import torch.nn as nn
from torchvision import models


class FaceDetectorResNet18(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.backbone(x)  # (B, 1)


def build_resnet18(dropout=0.5):
    return FaceDetectorResNet18(dropout=dropout)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params totaux : {total:,} | Entraînables : {trainable:,}")
    return {"total": total, "trainable": trainable}