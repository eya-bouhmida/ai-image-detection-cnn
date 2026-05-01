import os
import sys

# ── Fix OpenMP + Path ─────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Imports PyTorch ───────────────────────────────────────────
import torch

# ── Imports projet ────────────────────────────────────────────
from src.data_loaders.dataset import get_data_loaders
from src.models.model         import build_model, count_params
from src.models.train_utils   import train, plot_training_curves

# ── Configuration ─────────────────────────────────────────────
cfg = {
    # ── Données ──────────────────────────────────────────────
    "batch_size"          : 32,
    "num_workers"         : 2,

    # ── Modèle ───────────────────────────────────────────────
    "dropout"             : 0.5,

    # ── Optimiseur ───────────────────────────────────────────
    "lr"                  : 1e-3,
    "weight_decay"        : 1e-4,

    # ── Scheduler ────────────────────────────────────────────
    "lr_patience"         : 3,
    "lr_factor"           : 0.5,

    # ── Entraînement ─────────────────────────────────────────
    "epochs"              : 30,
    "early_stop_patience" : 7,

    # ── Sauvegarde ───────────────────────────────────────────
    "save_dir"            : "output",
    "experiment"          : "exp_01",
}

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # 1. Données
    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    # 2. Modèle
    model = build_model(dropout=cfg["dropout"]).to(device)
    count_params(model)

    # 3. Entraînement
    model, history = train(cfg, model, train_loader, val_loader, device)

    # 4. Courbes
    plot_training_curves(history)