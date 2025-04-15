import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def get_dataloaders(data_dir, batch_size=32, split_ratio=0.8, transform=None):
    if transform is None:
        raise ValueError("Transform must be provided (with or without augmentations).")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    targets = np.array(dataset.targets)

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=1 - split_ratio, random_state=42)
    for train_idx, val_idx in strat_split.split(np.zeros(len(targets)), targets):
        train_data = torch.utils.data.Subset(dataset, train_idx)
        val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(dataset.classes)
