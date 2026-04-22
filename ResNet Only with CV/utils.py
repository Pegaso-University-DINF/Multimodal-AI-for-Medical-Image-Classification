import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

def evaluate(model, dataloader, device, num_classes):
    """
    Evaluate the model and return multiple metrics:
    accuracy, precision, recall, f1-score, ROC-AUC, PR-AUC
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # Metrics
    acc = accuracy_score(all_labels_np, all_preds_np)
    precision = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

    # Binarize labels for multi-class ROC-AUC & PR-AUC
    labels_onehot = label_binarize(all_labels_np, classes=list(range(num_classes)))
    
    try:
        roc_auc = roc_auc_score(labels_onehot, all_probs_np, average='weighted', multi_class='ovr')
    except ValueError:
        roc_auc = float('nan')  # Not defined if one class missing

    try:
        pr_auc = average_precision_score(labels_onehot, all_probs_np, average='weighted')
    except ValueError:
        pr_auc = float('nan')

    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    return metrics
