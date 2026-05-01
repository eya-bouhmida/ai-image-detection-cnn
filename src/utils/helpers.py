from importlib.resources import path
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.utils.consts import OUTPUT_DIR, CLASS_NAMES
import json

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"✅ Modèle sauvegardé : {path}")

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    print(f"✅ Modèle chargé : {path}")
    return model

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(train_accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.legend()

    plt.savefig(OUTPUT_DIR + "training_curves.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES, 
                cmap="Blues")
    plt.title("Matrice de confusion")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.savefig(OUTPUT_DIR + "confusion_matrix.png")
    plt.show()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    
def save_json(data, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Sauvegardé : {path}")