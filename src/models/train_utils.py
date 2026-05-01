import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.helpers import save_json


def run_epoch(model, loader, device, optimizer=None):
    """
    Un passage complet sur loader.
    optimizer fourni  → mode entraînement
    optimizer = None  → mode évaluation (pas de gradient)
    Retourne un dict avec loss et accuracy.
    """
    is_train  = optimizer is not None
    criterion = nn.BCEWithLogitsLoss()   # sigmoid intégré, numériquement stable

    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels, *_ in tqdm(loader, leave=False):
            imgs     = imgs.to(device)
            labels_f = labels.float().to(device)   # float pour BCEWithLogitsLoss
            labels_l = labels.long().to(device)    # long  pour l'accuracy

            logits   = model(imgs).squeeze(1)      # (B,) logit brut

            loss = criterion(logits, labels_f)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Prédictions via seuil 0.5
            preds    = (torch.sigmoid(logits) >= 0.5).long()
            correct  += (preds == labels_l).sum().item()

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            n_samples  += bs

    return {
        "loss": total_loss / n_samples,
        "acc" : correct    / n_samples,
    }


def train(cfg, model, train_loader, val_loader, device):
    print(f"Device : {device}\n")
    os.makedirs(cfg["save_dir"], exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg["lr_patience"], factor=cfg["lr_factor"]
    )

    best_val_loss    = float("inf")
    no_improve_count = 0
    history          = {"train": [], "valid": []}

    for epoch in range(1, cfg["epochs"] + 1):
        train_m = run_epoch(model, train_loader, device, optimizer)
        val_m   = run_epoch(model, val_loader,   device)

        scheduler.step(val_m["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train"].append(train_m)
        history["valid"].append(val_m)

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']}  "
            f"| train loss={train_m['loss']:.4f}  acc={train_m['acc']:.3f}  "
            f"| val   loss={val_m['loss']:.4f}  acc={val_m['acc']:.3f}  "
            f"| lr={current_lr:.2e}"
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss    = val_m["loss"]
            no_improve_count = 0

            ckpt_path = os.path.join(cfg["save_dir"], f"{cfg['experiment']}.pt")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : best_val_loss,
            }, ckpt_path)
            print(f"  ✓ Checkpoint sauvegardé : {ckpt_path}  (val_loss={best_val_loss:.4f})")

        else:
            no_improve_count += 1
            if no_improve_count >= cfg["early_stop_patience"]:
                print(f"\nEarly stopping — pas d'amélioration depuis {cfg['early_stop_patience']} époques.")
                break

    save_json(cfg,     os.path.join(cfg["save_dir"], f"{cfg['experiment']}_config.json"))
    save_json(history, os.path.join(cfg["save_dir"], f"{cfg['experiment']}_history.json"))

    print(f"\nEntraînement terminé. Meilleur val_loss : {best_val_loss:.4f}")
    return model, history


def plot_training_curves(history):
    """
    history = {"train": [...], "valid": [...]}
    Chaque élément : {"loss": ..., "acc": ...}
    """
    train_h = history["train"]
    val_h   = history["valid"]
    epochs  = range(1, len(train_h) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key in zip(axes, ["loss", "acc"]):
        ax.plot(epochs, [m[key] for m in train_h], label="train")
        ax.plot(epochs, [m[key] for m in val_h],   label="val", linestyle="--")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()