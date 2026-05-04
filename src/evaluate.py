import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.data_loaders.dataset import get_data_loaders
from src.models.model import build_model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    cfg = {"batch_size": 32, "num_workers": 0}
    _, _, test_loader = get_data_loaders(cfg)

    model = build_model()
    checkpoint = torch.load("output/exp_01.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, *_ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["FAKE", "REAL"],
                yticklabels=["FAKE", "REAL"],
                cmap="Blues")
    plt.title("Matrice de confusion — Test set (20 000 images)", fontweight="bold")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.savefig("output/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n" + classification_report(all_labels, all_preds,
                                       target_names=["FAKE", "REAL"]))
    print("Sauvegarde : output/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()