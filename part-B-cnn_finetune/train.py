import os
import sys
import wandb
import torch
import random
import logging
import argparse
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from datasetloader import get_dataloaders  # Import custom dataloader

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Training function
def train(config, model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader, save_model=False):
    # WandB initialization and custom run name
    if config.use_wandb and not wandb.run:
        wandb.init(project=config.wandb_project, config=config)

    if config.use_wandb:
        wandb.run.name = f"dense{config.dense_size}_drop{config.dropout}_freeze{config.freeze_option}"
        wandb.run.save()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    best_valid_acc = 0.0
    best_model_path = None

    for epoch in range(config.n_epochs):
        model.train()
        train_acc = 0
        train_loss = []
        cnt = 0
        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_acc += (torch.argmax(y_pred, 1) == yb).float().sum()
            cnt += len(yb)
            train_loss.append(loss.item())
        train_acc /= cnt
        train_loss = np.mean(train_loss)

        model.eval()
        valid_acc = 0
        valid_loss = []
        cnt = 0
        with torch.no_grad():
            for xb, yb in valid_dataloader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = loss_fn(y_pred, yb)
                valid_acc += (torch.argmax(y_pred, 1) == yb).float().sum()
                cnt += len(yb)
                valid_loss.append(loss.item())
            valid_acc /= cnt
            valid_loss = np.mean(valid_loss)

        scheduler.step(valid_loss)

        print(f"Epoch {epoch+1}: valid accuracy {valid_acc*100:.2f}%, train accuracy {train_acc*100:.2f}%, train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

        if config.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc * 100,
                'valid_acc': valid_acc * 100,
            })

        if save_model and valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_path = f"best_model_{wandb.run.id}.pth"
            torch.save(model.state_dict(), best_model_path)

    if best_model_path:
        print(f"\nBest model saved to {best_model_path} with validation accuracy: {best_valid_acc*100:.2f}%")

    torch.cuda.empty_cache()

# Sweep function
def wandb_sweep():
    with wandb.init() as run:
        config = wandb.config

        # Custom run name in sweep mode
        run.name = f"sweep_dense{config.dense_size}_drop{config.dropout}_freeze{config.freeze_option}"
        run.save()

        train_loader, val_loader, classes = get_dataloaders(
            data_dir=config.dataset_path,
            batch_size=config.batch_size,
            input_size=config.in_dims,
            val_split=0.2
        )

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_feats, config.dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.dense_size, len(classes))
        )

        for param in model.parameters():
            param.requires_grad = False

        if config.freeze_option == 0:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif config.freeze_option == 1:
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.99), weight_decay=config.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

        train(config, model, loss_fn, optimizer, scheduler, train_loader, val_loader, save_model=True)

# Main function
def main(args: argparse.Namespace):
    if args.use_wandb:
        wandb.login()
        sweep_config = {
            'method': 'bayes',
            'name': 'CNN finetune',
            'metric': {'name': 'valid_acc', 'goal': 'maximize'},
            'parameters': {
                'dataset_path': {'value': args.dataset_path},
                'in_dims': {'values': [224]},
                'n_epochs': {'values': [10, 15]},
                'learning_rate': {'values': [1e-3, 1e-4]},
                'weight_decay': {'values': [0, 0.005, 0.5]},
                'batch_size': {'values': [64, 128]},
                'dense_size': {'values': [256, 512, 1024]},
                'dropout': {'values': [0, 0.2, 0.5]},
                'freeze_option': {'values': [0, 1]},
                'wandb_project': {'value': args.wandb_project},
                'use_wandb': {'value': True},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=wandb_sweep, count=30)
        wandb.finish()
    else:
        if args.use_wandb and not wandb.run:
            wandb.init(project=args.wandb_project, config=args)
            wandb.run.name = f"manual_dense{args.dense_size}_drop{args.dropout}_freeze{args.freeze_option}"
            wandb.run.save()

        # Manual training logic
        train_loader, val_loader, classes = get_dataloaders(
            data_dir=args.dataset_path,
            batch_size=args.batch_size,
            input_size=args.in_dims,
            val_split=0.2
        )

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_feats = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_feats, args.dense_size),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
            nn.Linear(args.dense_size, len(classes))
        )

        for param in model.parameters():
            param.requires_grad = False

        if args.freeze_option == 0:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif args.freeze_option == 1:
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

        # Pass the args directly to the train function for manual runs
        train(args, model, loss_fn, optimizer, scheduler, train_loader, val_loader, save_model=True)

# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-uw", "--use_wandb", default=True, action="store_true", help="Use Weights and Biases or not")
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment2", help="Project name used in Weights & Biases")
    parser.add_argument("-dp", "--dataset_path", type=str, default="/home/malavika/da6401_assignment2/inaturalist_12K/train", help="Path to dataset")
    parser.add_argument("-in", "--in_dims", type=int, default=256, help="Input image dimensions")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-ds", "--dense_size", type=int, default=512, help="Dense layer size")
    parser.add_argument("-do", "--dropout", type=float, default=0, help="Dropout rate for dense layer")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.005, help="Weight decay for optimizer")
    parser.add_argument("-ne", "--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("-fr", "--freeze_option", type=int, default=0, choices=[0, 1], help="Freeze options")

    args = parser.parse_args()
    main(args)
