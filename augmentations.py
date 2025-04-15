from torchvision import transforms

def get_transforms(augmentations=True):
    if augmentations:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    return transform
