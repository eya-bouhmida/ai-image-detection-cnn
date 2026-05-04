import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from src.data_loaders.dataset import get_data_loaders
from src.models.model_resnet import build_resnet18

def evaluate_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    cfg = {"batch_size": 32, "num_workers": 2}
    _, _, test_loader = get_data_loaders(cfg)

    model = build_resnet18()
    checkpoint = torch.load("output/exp_02_resnet18.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print("Modèle ResNet18 chargé !")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, *_ in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    report   = classification_report(all_labels, all_preds,
                                      target_names=["FAKE", "REAL"], output_dict=True)

    print(f"\nAccuracy ResNet18 : {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))

    # ── Visualisations ─────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Évaluation ResNet18 — 20 000 images test", fontsize=16, fontweight="bold")

    # 1. Matrice de confusion
    ax1 = fig.add_subplot(2, 3, 1)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["FAKE", "REAL"],
                yticklabels=["FAKE", "REAL"], ax=ax1)
    ax1.set_title("Matrice de confusion", fontweight="bold")
    ax1.set_ylabel("Vrai label")
    ax1.set_xlabel("Prédit")

    # 2. Courbe ROC
    ax2 = fig.add_subplot(2, 3, 2)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc     = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax2.set_title("Courbe ROC", fontweight="bold")
    ax2.set_xlabel("Taux faux positifs")
    ax2.set_ylabel("Taux vrais positifs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Distribution des probabilités
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(all_probs[all_labels == 0], bins=50, alpha=0.7,
             color="red", label="FAKE", density=True)
    ax3.hist(all_probs[all_labels == 1], bins=50, alpha=0.7,
             color="green", label="REAL", density=True)
    ax3.axvline(x=0.5, color="black", linestyle="--", label="Seuil 0.5")
    ax3.set_title("Distribution des probabilités", fontweight="bold")
    ax3.set_xlabel("Probabilité prédite")
    ax3.set_ylabel("Densité")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Métriques par classe
    ax4 = fig.add_subplot(2, 3, 4)
    classes   = ["FAKE", "REAL"]
    precision = [report["FAKE"]["precision"], report["REAL"]["precision"]]
    recall    = [report["FAKE"]["recall"],    report["REAL"]["recall"]]
    f1        = [report["FAKE"]["f1-score"],  report["REAL"]["f1-score"]]
    x         = np.arange(len(classes))
    width     = 0.25
    ax4.bar(x - width, precision, width, label="Precision", color="#9B59B6")
    ax4.bar(x,         recall,    width, label="Recall",    color="#8E44AD")
    ax4.bar(x + width, f1,        width, label="F1-score",  color="#6C3483")
    ax4.set_title("Métriques par classe", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.set_ylim([0.95, 1.01])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax4.text(i - width, p + 0.001, f"{p:.3f}", ha="center", fontsize=8)
        ax4.text(i,         r + 0.001, f"{r:.3f}", ha="center", fontsize=8)
        ax4.text(i + width, f + 0.001, f"{f:.3f}", ha="center", fontsize=8)

    # 5. Tableau récapitulatif
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis("off")
    data = [
        ["Modèle",            "ResNet18"],
        ["Accuracy",          f"{accuracy:.2f}%"],
        ["AUC-ROC",           f"{roc_auc:.4f}"],
        ["Precision FAKE",    f"{report['FAKE']['precision']*100:.2f}%"],
        ["Recall FAKE",       f"{report['FAKE']['recall']*100:.2f}%"],
        ["F1-score FAKE",     f"{report['FAKE']['f1-score']*100:.2f}%"],
        ["Precision REAL",    f"{report['REAL']['precision']*100:.2f}%"],
        ["Recall REAL",       f"{report['REAL']['recall']*100:.2f}%"],
        ["F1-score REAL",     f"{report['REAL']['f1-score']*100:.2f}%"],
        ["Total images test", "20 000"],
    ]
    table = ax5.table(cellText=data, colLabels=["Métrique", "Valeur"],
                      cellLoc="center", loc="center", colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#9B59B6")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F4ECF7")
    ax5.set_title("Résumé ResNet18", fontweight="bold", pad=20)

    # 6. Détail prédictions
    ax6 = fig.add_subplot(2, 3, 6)
    tn, fp, fn, tp = cm.ravel()
    categories = ["Vrais\nFAKE (TN)", "Faux\nREAL (FP)", "Faux\nFAKE (FN)", "Vrais\nREAL (TP)"]
    values     = [tn, fp, fn, tp]
    colors_bar = ["#4CAF50", "#F44336", "#F44336", "#4CAF50"]
    bars = ax6.bar(categories, values, color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax6.set_title("Détail des prédictions", fontweight="bold")
    ax6.set_ylabel("Nombre d'images")
    ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("output/evaluation_resnet18.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("Sauvegardé : output/evaluation_resnet18.png")

    return accuracy, roc_auc, report

if __name__ == "__main__":
    evaluate_resnet()