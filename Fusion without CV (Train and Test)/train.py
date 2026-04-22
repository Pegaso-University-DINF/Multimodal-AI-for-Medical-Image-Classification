import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate

def train_model(model, train_loader, val_loader, device, num_classes, num_epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, input_ids, attention_mask, labels in train_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        metrics = evaluate(model, val_loader, device, num_classes)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss/len(train_loader):.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | "
              f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    return model
