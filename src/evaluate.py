import torch
from src.utils.consts import *
from src.utils.helpers import load_checkpoint, plot_confusion_matrix
from src.data_loaders.dataset import load_data
from src.models.model import build_model

def evaluate():
    cfg = {"batch_size": 32, "num_workers": 2}
    _, _, test_loader = load_data(cfg)
    model = build_model()
    model = load_checkpoint(model, MODEL_SAVE_PATH)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    evaluate()