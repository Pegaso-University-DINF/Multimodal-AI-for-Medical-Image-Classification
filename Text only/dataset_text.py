# import torch
# from torch.utils.data import Dataset
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from transformers import AutoTokenizer

# class TextDataset(Dataset):
#     def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_len=256):
#         self.data = pd.read_csv(csv_file)
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.max_len = max_len
        
#         # Encode labels
#         self.label_encoder = LabelEncoder()
#         self.data['label_encoded'] = self.label_encoder.fit_transform(self.data['label'])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         caption = str(self.data.iloc[idx]['caption'])
#         label = self.data.iloc[idx]['label_encoded']

#         encoding = self.tokenizer(
#             caption,
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_len,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': encoding['input_ids'].squeeze(0),
#             'attention_mask': encoding['attention_mask'].squeeze(0),
#             'label': torch.tensor(label, dtype=torch.long)
#         }


# dataset_text.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd


class TextOnlyDataset(Dataset):
    def __init__(self,
                 csv_file,
                 tokenizer_name='bert-base-uncased',
                 max_length=128):

        self.df = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Label mapping
        self.label2idx = {
            label: idx
            for idx, label in enumerate(sorted(self.df['label'].unique()))
        }

        self.num_classes = len(self.label2idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = row['caption']
        label = self.label2idx[row['label']]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label


def get_text_loaders(csv_file,
                     batch_size=16,
                     val_split=0.2):

    dataset = TextOnlyDataset(csv_file)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, val_loader, dataset.num_classes

