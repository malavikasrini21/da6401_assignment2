# datasetloader.py
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from augmentations import get_train_transforms, get_test_transforms

def get_dataloaders(data_dir, batch_size=32, input_size=256, val_split=0.2):
    full_dataset = datasets.ImageFolder(data_dir)
    targets = [sample[1] for sample in full_dataset.samples]
    
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=targets,
        random_state=42
    )

    train_transform = get_train_transforms(input_size)
    test_transform = get_test_transforms(input_size)

    # Update the dataset with transformations
    full_dataset.transform = train_transform
    train_dataset = Subset(full_dataset, train_idx)

    full_dataset.transform = test_transform
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, full_dataset.classes
