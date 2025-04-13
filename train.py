import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import CNNModel
from datasetloader import get_dataloaders
import wandb
import argparse

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, num_classes = get_dataloaders(config.data_dir, config.batch_size)

        model = CNNModel(
            input_channels=3,
            num_classes=num_classes,
            num_layers=config.num_layers,
            filters=config.filters,
            kernel_size=config.kernel_size,
            activation_fn=getattr(nn, config.activation),
            dense_neurons=config.dense_neurons
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            correct, total = 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            val_acc = evaluate(model, val_loader, device)
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "val_acc": val_acc
            })

        torch.save(model.state_dict(), f"{config.model_save_path}/cnn_model.pth")

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

    # Only used in non-sweep runs
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--filters", nargs='+', type=int, default=[32])
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--activation", type=str, default="ReLU")
    parser.add_argument("--dense_neurons", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if args.sweep:
        sweep_config = {
            'method': 'grid',
            'name': 'CNN Sweep',
            'metric': {
                'name': 'val_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {'values': [5, 10, 15]},
                'batch_size': {'values': [32, 64]},
                'num_layers': {'values': [5]},
                'filters': {'values': [32, 64, 128]},
                'kernel_size': {'values': [7,5,5,3,3]},
                'activation': {'values': ['ReLU']},
                'dense_neurons': {'values': [128, 256, 512]},
                'lr': {'values': [0.001, 0.0001]},
                'data_dir': {'value': args.data_dir},
                'model_save_path': {'value': args.model_save_path}
            }
        }

        sweep_id = wandb.sweep(sweep_config, project="da6401_assignment1")
        wandb.agent(sweep_id, function=train, count=50)
    else:
        # Manual run without sweep
        wandb.init(project="da6401_assignment1")
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_layers": args.num_layers,
            "filters": args.filters,
            "kernel_size": args.kernel_size,
            "activation": args.activation,
            "dense_neurons": args.dense_neurons,
            "lr": args.lr,
            "data_dir": args.data_dir,
            "model_save_path": args.model_save_path
        }
        train(config)
