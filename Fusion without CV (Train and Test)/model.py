# import torch
# import torch.nn as nn
# from torchvision import models
# from transformers import AutoModel, AutoConfig

# class MultimodalClassifier(nn.Module):
#     def __init__(self, num_classes, text_model_name='bert-base-uncased', image_model_name='resnet50', text_hidden_dim=768):
#         super(MultimodalClassifier, self).__init__()
        
#         # --- Image Encoder ---
#         if image_model_name == 'resnet50':
#             self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         elif image_model_name == 'resnet18':
#             self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         elif image_model_name == 'resnet34':
#             self.image_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#         elif image_model_name == 'resnet101':
#             self.image_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

#         else:
#             raise ValueError("Unsupported image model")
#         self.image_feat_dim = self.image_model.fc.in_features
#         self.image_model.fc = nn.Identity()  # remove final classifier

#         # --- Text Encoder ---
#         self.text_model = AutoModel.from_pretrained(text_model_name)
#         self.text_feat_dim = text_hidden_dim

#         # --- Fusion + Classifier ---
#         fusion_dim = self.image_feat_dim + self.text_feat_dim
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, image, input_ids, attention_mask):
#         # --- Image features ---
#         img_feat = self.image_model(image)

#         # --- Text features ---
#         text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
#         text_feat = text_outputs.last_hidden_state[:,0,:]  # CLS token

#         # --- Fusion ---
#         fused_feat = torch.cat((img_feat, text_feat), dim=1)
#         out = self.classifier(fused_feat)
#         return out

#++++++++++++++++++++++Modality Attention+++++++++++++++++++++++
# import torch
# import torch.nn as nn
# from torchvision import models
# from transformers import AutoModel


# class MultimodalClassifier(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  text_model_name='bert-base-uncased',
#                  text_hidden_dim=768):
#         super(MultimodalClassifier, self).__init__()

#         # -----------------------------
#         # Image Encoder (ResNet50 ONLY)
#         # -----------------------------
#         self.image_model = models.resnet50(
#             weights=models.ResNet50_Weights.DEFAULT
#         )
#         self.image_feat_dim = self.image_model.fc.in_features
#         self.image_model.fc = nn.Identity()  # remove classifier

#         # Project image features to same dim as text
#         self.image_proj = nn.Linear(self.image_feat_dim, text_hidden_dim)

#         # -----------------------------
#         # Text Encoder (BERT)
#         # -----------------------------
#         self.text_model = AutoModel.from_pretrained(text_model_name)
#         self.text_feat_dim = text_hidden_dim

#         # -----------------------------
#         # Modality Attention
#         # -----------------------------
#         self.attention_layer = nn.Linear(text_hidden_dim * 2, 2)

#         # -----------------------------
#         # Final Classifier
#         # -----------------------------
#         self.classifier = nn.Sequential(
#             nn.Linear(text_hidden_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, image, input_ids, attention_mask):

#         # ----- Image branch -----
#         img_feat = self.image_model(image)          # (B, 2048)
#         img_feat = self.image_proj(img_feat)       # (B, 768)

#         # ----- Text branch -----
#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS token (B, 768)

#         # ----- Modality Attention -----
#         combined = torch.cat((img_feat, text_feat), dim=1)   # (B, 1536)
#         attn_weights = self.attention_layer(combined)        # (B, 2)
#         attn_weights = torch.softmax(attn_weights, dim=1)

#         w_img = attn_weights[:, 0].unsqueeze(1)
#         w_txt = attn_weights[:, 1].unsqueeze(1)

#         fused_feat = w_img * img_feat + w_txt * text_feat    # (B, 768)

#         # ----- Classification -----
#         out = self.classifier(fused_feat)
#         return out


# +++++++++++ Gated Fusion +++++++++++
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

class MultimodalGatedClassifier(nn.Module):
    def __init__(self, num_classes, text_model_name='bert-base-uncased', text_hidden_dim=768):
        super(MultimodalGatedClassifier, self).__init__()

        # --- Image Encoder (ResNet50) ---
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_feat_dim = self.image_model.fc.in_features  # 2048
        self.image_model.fc = nn.Identity()

        # --- Text Encoder (BERT) ---
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = text_hidden_dim  # 768

        # --- Project text features to image feature dimension ---
        self.text_proj = nn.Linear(self.text_feat_dim, self.image_feat_dim)

        # --- Gated Fusion ---
        self.gate_img = nn.Sequential(
            nn.Linear(self.image_feat_dim, self.image_feat_dim),
            nn.Sigmoid()
        )
        self.gate_text = nn.Sequential(
            nn.Linear(self.image_feat_dim, self.image_feat_dim),
            nn.Sigmoid()
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # --- Image features ---
        img_feat = self.image_model(image)  # [batch, 2048]

        # --- Text features ---
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.last_hidden_state[:,0,:]  # CLS token [batch, 768]
        text_feat = self.text_proj(text_feat)  # [batch, 2048]

        # --- Apply Gated Fusion ---
        g_img = self.gate_img(img_feat)
        g_text = self.gate_text(text_feat)

        fused_feat = g_img * img_feat + g_text * text_feat  # [batch, 2048]

        # --- Classifier ---
        out = self.classifier(fused_feat)
        return out

