import torch
import torch.nn as nn


class FaceDetectorCNN(nn.Module):
    """
    CNN binaire : retourne un logit brut (B, 1).
    Appliquer sigmoid en dehors pour obtenir une probabilité.
    """
    def __init__(self, dropout=0.5):          # ✅ supprimé num_numbers (inutile)
        super().__init__()

        self.backbone = nn.Sequential(
            self._conv_block(3,   32),         # (B,  32, H/2,  W/2)
            self._conv_block(32,  64),         # (B,  64, H/4,  W/4)
            self._conv_block(64, 128),         # (B, 128, H/8,  W/8)
            self._conv_block(128, 256),        # (B, 256, H/16, W/16)
            nn.AdaptiveAvgPool2d((4, 4))       # (B, 256, 4, 4)
        )

        flat_dim = 256 * 4 * 4  # 4096

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),      nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(256, 1),        # logit brut → sigmoid appliqué dans la loss
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        feats = self.backbone(x)        # (B, 256, 4, 4)
        return self.classifier(feats)   # (B, 1)


def build_model(dropout=0.5):
    return FaceDetectorCNN(dropout=dropout)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params totaux : {total:,} | Entraînables : {trainable:,}")
    return {"total": total, "trainable": trainable}