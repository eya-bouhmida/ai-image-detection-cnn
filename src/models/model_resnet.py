import torch
import torch.nn as nn
from torchvision import models


class FaceDetectorResNet(nn.Module):
    """
    ResNet50 pré-entraîné sur ImageNet.
    On remplace la dernière couche pour la détection FAKE/REAL.
    """
    def __init__(self, dropout=0.5, freeze_backbone=False):
        super().__init__()

        # Charge ResNet50 pré-entraîné
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Gèle le backbone si demandé (transfer learning pur)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remplace la dernière couche (1000 classes → 1 logit)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)   # logit brut → sigmoid dans la loss
        )

    def forward(self, x):
        return self.backbone(x)  # (B, 1)


def build_resnet(dropout=0.5, freeze_backbone=False):
    return FaceDetectorResNet(dropout=dropout, freeze_backbone=freeze_backbone)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params totaux : {total:,} | Entraînables : {trainable:,}")
    return {"total": total, "trainable": trainable}