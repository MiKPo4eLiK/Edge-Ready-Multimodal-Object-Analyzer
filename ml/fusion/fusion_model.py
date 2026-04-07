import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel


class MultiModalModel(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        # image backbone
        self.image_model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT
        )

        # freeze image features
        for param in self.image_model.features.parameters():
            param.requires_grad = False

        image_features = self.image_model.classifier[1].in_features
        self.image_model.classifier = nn.Identity()

        # text backbone
        self.text_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # freeze text model
        for param in self.text_model.parameters():
            param.requires_grad = False

        text_features = 768  # DistilBERT hidden size

        # fusion head
        self.fc1 = nn.Linear(image_features + text_features, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, image, input_ids, attention_mask) -> None:

        # image branch
        img_feat = self.image_model(image)

        if img_feat.dim() == 4:
            img_feat = torch.flatten(img_feat, 1)

        # text branch
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden = text_outputs.last_hidden_state  # (B, seq_len, 768)

        # mean pooling with attention mask
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        text_feat = summed / counts

        # fusion
        combined = torch.cat((img_feat, text_feat), dim=1)

        # classifier
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
