"""
Transfer-learning model: ImageNet-pretrained ResNet18, final layer replaced
for binary classification (benign vs malignant).
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """
    Args:
        num_classes: number of output classes (2 for benign/malignant).
        freeze_backbone: if True, freezes all pretrained layers and only
            trains the new final layer (faster, less prone to overfitting
            on small datasets, but less flexible). If False, the whole
            network fine-tunes (slower, needs a lower learning rate, but
            typically better final accuracy given enough data).
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer. This new layer is always
    # trainable, regardless of freeze_backbone.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
