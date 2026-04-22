# import torch
# import torch.nn as nn
# from transformers import AutoModel

# class TextOnlyClassifier(nn.Module):
#     def __init__(self, num_classes, model_name="bert-base-uncased"):
#         super(TextOnlyClassifier, self).__init__()
        
#         self.bert = AutoModel.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(768, num_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
        
#         cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
#         x = self.dropout(cls_output)
#         logits = self.classifier(x)

#         return logits


# model_text.py
# model_text.py
import torch
import torch.nn as nn
from transformers import AutoModel


class TextOnlyClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 text_model_name='bert-base-uncased',
                 text_hidden_dim=768):
        super(TextOnlyClassifier, self).__init__()

        # Text Encoder (same as fusion model)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = text_hidden_dim

        # Classification Head (same depth as fusion)
        self.classifier = nn.Sequential(
            nn.Linear(self.text_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        text_feat = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(text_feat)

        return logits
