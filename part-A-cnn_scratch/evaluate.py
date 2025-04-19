import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

from cnn_model import ConvNeuralNet

def evaluate(model_path, test_dir, batch_size, run_id=None):
    run = wandb.init(project="da6401_assignment2", id=run_id, resume="allow")
    config = run.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    class_names = test_dataset.classes
    num_classes = len(class_names)

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

    # Load model
    state_dict = torch.load(model_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds, all_labels, all_images = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(inputs.cpu())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    wandb.log({"test_accuracy": accuracy})

    # Select 3 images per class
    class_counts = {i: 0 for i in range(num_classes)}
    selected_imgs, selected_preds, selected_refs = [], [], []

    for img, true, pred in zip(all_images, all_labels, all_preds):
        if class_counts[true] < 3:
            selected_imgs.append(img)
            selected_preds.append(pred)
            selected_refs.append(true)
            class_counts[true] += 1
        if sum(class_counts.values()) >= 30:
            break

    # 10x3 Grid Plot
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    for idx, ax in enumerate(axes.flat):
        if idx < len(selected_imgs):
            img = selected_imgs[idx].detach().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Ref: {class_names[selected_refs[idx]]}\nPred: {class_names[selected_preds[idx]]}', fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    fig.savefig("prediction_grid.png", bbox_inches="tight")  # ✅ Save locally
    wandb.log({"Prediction Grid": wandb.Image(fig)})
    plt.close(fig)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix')

    plt.tight_layout()
    fig_cm.savefig("confusion_matrix.png", bbox_inches="tight")  # ✅ Save locally
    wandb.log({"Confusion Matrix": wandb.Image(fig_cm)})
    plt.close(fig_cm)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--run_id', type=str, required=True, help="Best wandb run ID")
    args = parser.parse_args()
    evaluate(args.model_path, args.test_dir, args.batch_size, args.run_id)
