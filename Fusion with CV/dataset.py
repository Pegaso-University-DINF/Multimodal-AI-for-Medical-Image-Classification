import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='bert-base-uncased', max_length=128, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Map textual labels to numeric
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.num_classes = len(self.label2idx)

        # Convert labels column to numeric
        self.df['label_idx'] = self.df['label'].map(self.label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Image ----
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # ---- Text ----
        encoding = self.tokenizer(
            row['caption'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        label = torch.tensor(row['label_idx'], dtype=torch.long)

        return image, input_ids, attention_mask, label


def get_full_dataset(csv_file, tokenizer_name='bert-base-uncased'):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    dataset = MultimodalDataset(
        csv_file,
        tokenizer_name=tokenizer_name,
        transform=transform
    )

    labels = dataset.df['label_idx'].values

    return dataset, labels, dataset.num_classes


def get_fold_loaders(dataset, train_idx, val_idx, batch_size=8, num_workers=0):

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
