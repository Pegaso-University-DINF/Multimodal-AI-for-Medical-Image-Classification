import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from dataset import get_full_dataset, get_fold_loaders
from model import get_resnet_model
from train import train_model
from utils import evaluate

if __name__ == "__main__":

    # ---- Config ----
    data_dir = 'dataset/'
    batch_size = 8
    num_epochs = 10
    lr = 1e-4
    n_splits = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load full dataset ----
    full_dataset, labels, num_classes = get_full_dataset(data_dir)

    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=42)

    all_metrics = []

    # ---- 5-Fold Loop ----
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        print(f"\n========== Fold {fold+1}/{n_splits} ==========")

        train_loader, val_loader = get_fold_loaders(
            full_dataset,
            train_idx,
            val_idx,
            batch_size=batch_size
        )

        model = get_resnet_model(
            model_name='resnet101',
            num_classes=num_classes,
            pretrained=True
        )

        model = train_model(
            model,
            train_loader,
            val_loader,
            device,
            num_classes,
            num_epochs=num_epochs,
            lr=lr
        )

        metrics = evaluate(model, val_loader, device, num_classes)
        all_metrics.append(metrics)

    # ---- Compute Mean ± Std ----
    print("\n========== 5-Fold Cross Validation Results ==========")

    final_results = {}

    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        final_results[key] = (mean, std)

    for key, (mean, std) in final_results.items():
        print(f"{key}: {mean:.4f} ± {std:.4f}")
