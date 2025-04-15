import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import argparse
import os
from cnn_model import CNNModel  # import your CNN model

def evaluate(model_path, test_dir, batch_size):
    # Initialize the same run for logging evaluation
    run = wandb.init(project="da6401_assignment1", id="ft6q813w", resume="allow")

    # Load the best config
    config = run.config
    print(config)
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model with best sweep config
    model = CNNModel(
        input_channels=3,
        num_layers=config.num_layers,
        filters=config.filters,
        kernel_size=config.kernel_size,
        activation_fn=getattr(nn, config.activation),
        dense_neurons=config.dense_neurons,
        num_classes=10
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(model)
    model.to(device)
    model.eval()

    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Log to wandb
    wandb.log({"test_accuracy": accuracy})
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    evaluate(args.model_path, args.test_dir, args.batch_size)
