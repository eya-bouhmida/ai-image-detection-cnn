import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Chemins ──────────────────────────────────────────────────
BASE_DIR       = "data/real_vs_fake/real-vs-fake/"
TRAIN_FAKE_DIR = BASE_DIR + "train/fake/"
TRAIN_REAL_DIR = BASE_DIR + "train/real/"
VALID_FAKE_DIR = BASE_DIR + "valid/fake/"
VALID_REAL_DIR = BASE_DIR + "valid/real/"
TEST_FAKE_DIR  = BASE_DIR + "test/fake/"
TEST_REAL_DIR  = BASE_DIR + "test/real/"

IMG_SIZE   = 224
BATCH_SIZE = 32

# ── Dataset ───────────────────────────────────────────────────
class FaceDataset(Dataset):

    # ── INIT : collecte les chemins + labels ──────────────────
    def __init__(self, split="train"):
        self.split     = split
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # d'abord
            transforms.ToTensor(),                   # ensuite
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        if split == "train":
            folders = [(TRAIN_FAKE_DIR, 0), (TRAIN_REAL_DIR, 1)]
        elif split == "valid":
            folders = [(VALID_FAKE_DIR, 0), (VALID_REAL_DIR, 1)]
        elif split == "test":
            folders = [(TEST_FAKE_DIR, 0), (TEST_REAL_DIR, 1)]

        self.samples = []
        for folder, label in folders:
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "path"  : os.path.join(folder, fname),
                        "class" : label,
                        "split" : split
                    })

    def __len__(self):
        return len(self.samples)

    # ── GETITEM : charge l'image + retourne path, class, split ─
    def __getitem__(self, idx):
        item      = self.samples[idx]
        img_path  = item["path"]
        raw_class = item["class"]
        split     = item["split"]

        img        = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label      = torch.tensor(raw_class, dtype=torch.long)

        return img_tensor, label, split, img_path


# ── DataLoaders ───────────────────────────────────────────────
def get_data_loaders(cfg):
    train_ds = FaceDataset(split="train")
    valid_ds = FaceDataset(split="valid")
    test_ds  = FaceDataset(split="test")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"])
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)} | Test: {len(test_ds)}\n")
    return train_loader, valid_loader, test_loader


# ── TEST : afficher les 5 premiers éléments ───────────────────
if __name__ == "__main__":
    ds = FaceDataset(split="train")

    print(f"\n📦 Total images train : {len(ds)}\n")
    print("🔍 Les 5 premiers éléments :\n")

    for i in range(5):
        img_tensor, label, split, img_path = ds[i]
        print(f"[{i+1}] Path   : {img_path}")
        print(f"     Classe : {'🟥 FAKE' if label.item() == 0 else '✅ REAL'} (label={label.item()})")
        print(f"     Split  : {split}")
        print(f"     Shape  : {img_tensor.shape}")
        print()