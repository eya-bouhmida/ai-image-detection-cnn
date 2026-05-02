import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data_loaders.dataset import get_data_loaders
from src.models.model_resnet  import build_resnet, count_params
from src.models.train_utils   import train, plot_training_curves

cfg = {
    # ── Données ──────────────────────────────────────────────
    "batch_size"          : 32,
    "num_workers"         : 2,

    # ── Modèle ───────────────────────────────────────────────
    "dropout"             : 0.5,
    "freeze_backbone"     : False,  # True = transfer learning pur

    # ── Optimiseur ───────────────────────────────────────────
    "lr"                  : 1e-4,   # plus petit que CNN from scratch
    "weight_decay"        : 1e-4,

    # ── Scheduler ────────────────────────────────────────────
    "lr_patience"         : 3,
    "lr_factor"           : 0.5,

    # ── Entraînement ─────────────────────────────────────────
    "epochs"              : 20,
    "early_stop_patience" : 5,

    # ── Sauvegarde ───────────────────────────────────────────
    "save_dir"            : "output",
    "experiment"          : "exp_02_resnet",  # nouveau nom !
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    train_loader, val_loader, _ = get_data_loaders(cfg)

    model = build_resnet(
        dropout=cfg["dropout"],
        freeze_backbone=cfg["freeze_backbone"]
    ).to(device)
    count_params(model)

    model, history = train(cfg, model, train_loader, val_loader, device)
    plot_training_curves(history)