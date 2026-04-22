# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def train_model(model, train_loader, val_loader, device, epochs=5):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4)

#     model.to(device)

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         for batch in train_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

#     return model


# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     roc_auc_score,
#     average_precision_score
# )
# import torch.nn.functional as F
# import numpy as np


# def evaluate_model(model, loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for batch in loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)

#             outputs = model(input_ids, attention_mask)

#             # Convert logits to probabilities
#             probs = F.softmax(outputs, dim=1)

#             preds = torch.argmax(probs, dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())

#     all_labels = np.array(all_labels)
#     all_probs = np.array(all_probs)

#     # Basic metrics
#     acc = accuracy_score(all_labels, all_preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         all_labels, all_preds, average='weighted'
#     )

#     # Multi-class ROC-AUC (One-vs-Rest)
#     roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

#     # Multi-class PR-AUC
#     pr_auc = average_precision_score(
#         np.eye(all_probs.shape[1])[all_labels],
#         all_probs,
#         average='weighted'
#     )

#     print("\nText-Only Results:")
#     print(f"Accuracy:  {acc:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall:    {recall:.4f}")
#     print(f"F1 Score:  {f1:.4f}")
#     print(f"ROC-AUC:   {roc_auc:.4f}")
#     print(f"PR-AUC:    {pr_auc:.4f}")

#     return acc, precision, recall, f1, roc_auc, pr_auc


# train.py
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for input_ids, attention_mask, labels in tqdm(loader):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(loader), acc, precision, recall, f1


def evaluate(model, loader, criterion, device, num_classes):

    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    labels_binarized = label_binarize(all_labels, classes=range(num_classes))

    roc_auc = roc_auc_score(
        labels_binarized,
        all_probs,
        average="macro",
        multi_class="ovr"
    )

    pr_auc = average_precision_score(
        labels_binarized,
        all_probs,
        average="macro"
    )

    return (
        total_loss / len(loader),
        acc,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    )
