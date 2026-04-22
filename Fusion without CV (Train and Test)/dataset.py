# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from torchvision import transforms
# from transformers import AutoTokenizer
# import pandas as pd
# from sklearn.model_selection import train_test_split

# class MultimodalDataset(Dataset):
#     def __init__(self, csv_file, tokenizer_name='bert-base-uncased', max_length=128, transform=None):
#         """
#         csv_file: CSV with columns: image_path,label,caption
#         tokenizer_name: pretrained tokenizer for text
#         """
#         self.df = pd.read_csv(csv_file)
#         self.transform = transform
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_length = max_length

#         # Map textual labels to numeric
#         self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
#         self.num_classes = len(self.label2idx)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         # --- Image ---
#         image = Image.open(row['image_path']).convert('RGB')
#         if self.transform:
#             image = self.transform(image)

#         # --- Text ---
#         text = row['caption']
#         encoding = self.tokenizer(
#             text,
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         input_ids = encoding['input_ids'].squeeze(0)
#         attention_mask = encoding['attention_mask'].squeeze(0)

#         # --- Label ---
#         label = self.label2idx[row['label']]

#         return image, input_ids, attention_mask, label

# def get_loaders(csv_file, batch_size=16, val_split=0.2, num_workers=0, tokenizer_name='bert-base-uncased'):
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])

#     dataset = MultimodalDataset(csv_file, tokenizer_name=tokenizer_name, transform=transform)

#     val_size = int(len(dataset) * val_split)
#     train_size = len(dataset) - val_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader, val_loader, dataset.num_classes


#gated fusion 
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import pandas as pd

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='bert-base-uncased', max_length=128, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label2idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.num_classes = len(self.label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        image = Image.open(row['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Text
        text = row['caption']
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Label
        label = self.label2idx[row['label']]

        return image, input_ids, attention_mask, label

def get_loaders(csv_file, batch_size=16, val_split=0.2, num_workers=0, tokenizer_name='bert-base-uncased'):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = MultimodalDataset(csv_file, tokenizer_name=tokenizer_name, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, dataset.num_classes
