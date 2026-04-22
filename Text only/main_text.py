# main.py
import torch
import torch.nn as nn
import torch.optim as optim

from model_text import TextOnlyClassifier
from dataset_text import get_text_loaders
from train_text import train_one_epoch, evaluate


def main():

    # =====================
    # Configuration
    # =====================
    csv_file = "multimodal_dataset.csv"
    batch_size = 8
    epochs = 10
    learning_rate = 1e-4
    val_split = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # =====================
    # Data
    # =====================
    train_loader, val_loader, num_classes = get_text_loaders(
        csv_file,
        batch_size=batch_size,
        val_split=val_split
    )

    # =====================
    # Model
    # =====================
    model = TextOnlyClassifier(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # =====================
    # Training Loop
    # =====================
    for epoch in range(epochs):

        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        (val_loss,
         val_acc,
         val_prec,
         val_rec,
         val_f1,
         val_roc,
         val_pr) = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes
        )

        print(f"\nEpoch [{epoch+1}/{epochs}]")

        print("\nTrain Metrics:")
        print(f" Loss: {train_loss:.4f}")
        print(f" Accuracy: {train_acc:.4f}")
        print(f" Precision: {train_prec:.4f}")
        print(f" Recall: {train_rec:.4f}")
        print(f" F1-score: {train_f1:.4f}")

        print("\nValidation Metrics:")
        print(f" Loss: {val_loss:.4f}")
        print(f" Accuracy: {val_acc:.4f}")
        print(f" Precision: {val_prec:.4f}")
        print(f" Recall: {val_rec:.4f}")
        print(f" F1-score: {val_f1:.4f}")
        print(f" ROC-AUC: {val_roc:.4f}")
        print(f" PR-AUC: {val_pr:.4f}")


if __name__ == "__main__":
    main()

