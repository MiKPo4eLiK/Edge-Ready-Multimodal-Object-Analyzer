import torch.nn as nn
from torchvision import models


def build_model(num_classes: int):
    model = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT
    )

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model
