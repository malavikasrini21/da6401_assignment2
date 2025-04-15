import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import ConvNeuralNet
from datasetloader import get_dataloaders
from augmentations import get_transforms
import wandb
import argparse

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name=f"bs_{config.batch_size}_ca_{config.conv_activation}_fs_{config.kernel_size}_nf_{config.n_filters}_fo_{config.filter_org}_ep_{config.epochs}"
        wandb.run.name=run_name

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = get_transforms(config.augmentations)


        train_loader, val_loader, num_classes = get_dataloaders(
            config.data_dir, batch_size=config.batch_size, transform=transform
        )

        model = ConvNeuralNet(
            in_dims=(3, 256, 256),
            out_dims=num_classes,
            conv_activation=config.conv_activation,
            dense_activation=config.dense_activation,
            dense_size=config.dense_neurons,
            filter_size=config.kernel_size,
            n_filters=config.n_filters,
            filter_org=config.filter_org,
            batch_norm=config.batch_norm,
            dropout=config.dropout
        )

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        best_val_acc=0.0
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            val_acc = evaluate(model, val_loader, device)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader),
                "val_acc": val_acc
            })

        #torch.save(model.state_dict(), f"{config.model_save_path}/cnn_model.pth")
        if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{config.model_save_path}/best_model.pth")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, default=".")
    parser.add_argument("--sweep", action="store_true")

    # Manual run hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_filters", type=int, default=32)
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[7, 5, 5, 3, 3])
    parser.add_argument("--conv_activation", type=str, default="relu")
    parser.add_argument("--dense_activation", type=str, default="relu")
    parser.add_argument("--dense_neurons", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_norm", type=bool, default=True)
    parser.add_argument("--filter_org", type=str, default="same")
    parser.add_argument("--augmentations", type=bool, default=True)

    args = parser.parse_args()

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
            'name': 'CNN Sweep',
            'metric': {
                'name': 'val_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {'values': [10, 15]},
                'batch_size': {'values': [32, 64]},
                'n_filters': {'values': [32, 64,128]},
                'kernel_size': {'values': [[7,5,5,3,3], [3,3,5,5,7], [3,3,3,3,3], [5,5,5,5,5], [7,7,7,7,7]]},
                'conv_activation': {'values': ['relu', 'gelu','silu','mish']},
                'dense_activation': {'values': ['relu']},
                'dense_neurons': {'values': [128, 256, 512,1024]},
                'lr': {'values': [0.001, 0.0001]},
                'dropout': {'values': [0,0.2, 0.3]},
                'batch_norm': {'values': [True]},
                'filter_org': {'values': ['same', 'double', 'halve']},
                'augmentations': {'values': [True, False]},
                'data_dir': {'value': args.data_dir},
                'model_save_path': {'value': args.model_save_path}
            }
        }

        sweep_id = wandb.sweep(sweep_config, project="da6401_assignment2")
        wandb.agent(sweep_id, function=train, count=25)
    else:
        
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_filters": args.n_filters,
            "kernel_size": args.kernel_size,
            "conv_activation": args.conv_activation,
            "dense_activation": args.dense_activation,
            "dense_neurons": args.dense_neurons,
            "lr": args.lr,
            "dropout": args.dropout,
            "batch_norm": args.batch_norm,
            "filter_org": args.filter_org,
            "data_dir": args.data_dir,
            "model_save_path": args.model_save_path
        }
        train(config)
