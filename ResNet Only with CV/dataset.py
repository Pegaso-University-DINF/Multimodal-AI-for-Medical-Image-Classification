import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
import numpy as np

def get_full_dataset(data_dir):
    """
    Returns full ImageFolder dataset and labels
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)

    return dataset, labels, num_classes


def get_fold_loaders(dataset, train_idx, val_idx, batch_size=32, num_workers=0):
    """
    Create train and validation loaders for a fold
    """
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_subset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader
