# 🔍 AI Face Detection — CNN vs ResNet18

> Détection de visages générés par Intelligence Artificielle (StyleGAN) à l'aide de réseaux de neurones convolutifs.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.31%25-brightgreen)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Description

Ce projet implémente et compare deux modèles de deep learning pour détecter si un visage est **réel** ou **généré par une IA (StyleGAN)** :

- **FaceDetectorCNN** — CNN custom entraîné from scratch (2.6M paramètres)
- **FaceDetectorResNet18** — ResNet18 pré-entraîné sur ImageNet (11M paramètres)

Un dashboard interactif (Gradio) permet de tester les modèles en temps réel en uploadant n'importe quelle image.

---

## 📁 Structure du projet

```
ai-image-detection-cnn/
│
├── data/
│   └── real_vs_fake/
│       └── real-vs-fake/
│           ├── train/               ← 100 000 images d'entraînement
│           ├── valid/               ← 20 000 images de validation
│           └── test/                ← 20 000 images de test
│
├── output/
│   ├── exp_01.pt                    ← Poids du modèle CNN sauvegardés
│   ├── exp_01_config.json           ← Configuration d'entraînement CNN
│   ├── exp_01_history.json          ← Historique loss/accuracy CNN
│   ├── exp_02_resnet18.pt           ← Poids du modèle ResNet18 sauvegardés
│   ├── exp_02_resnet18_config.json  ← Configuration d'entraînement ResNet18
│   ├── exp_02_resnet18_history.json ← Historique loss/accuracy ResNet18
│   ├── evaluation_complete.png      ← Visualisations complètes évaluation CNN
│   ├── confusion_matrix.png         ← Matrice de confusion CNN
│   └── visualisations_finales.png   ← Courbes d'entraînement CNN
│
├── src/
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   └── dataset.py               ← FaceDataset + get_data_loaders()
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py                 ← Architecture FaceDetectorCNN
│   │   ├── model_resnet.py          ← Architecture FaceDetectorResNet18
│   │   └── train_utils.py           ← Boucle d'entraînement + callbacks
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py               ← Fonctions utilitaires (save, load, plot)
│   │   └── consts.py                ← Constantes globales et chemins
│   ├── __init__.py
│   ├── train.py                     ← Script d'entraînement CNN (exp_01)
│   ├── train_resnet.py              ← Script d'entraînement ResNet18 (exp_02)
│   └── evaluate.py                  ← Évaluation finale sur test set
│
├── dashboard.py                     ← Interface Gradio interactive
├── .gitignore
├── environnement.env
└── README.md
```

---

## 📊 Dataset

| Propriété | Valeur |
|-----------|--------|
| Nom | [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| Source | Kaggle |
| Total images | 140 000 |
| Classes | REAL (70 000) / FAKE (70 000) |
| Équilibre | Parfaitement équilibré (50% / 50%) |
| Images FAKE | Générées par **StyleGAN** |
| Résolution | 224 × 224 pixels |
| Format | JPEG |

### Répartition des données

| Split | Images | Pourcentage |
|-------|--------|-------------|
| Train | 100 000 | 71.4% |
| Validation | 20 000 | 14.3% |
| Test | 20 000 | 14.3% |

### Préparation des données

Chaque image passe par les transformations suivantes avant d'entrer dans le modèle :

```python
transforms.Compose([
    transforms.Resize((224, 224)),          # Redimensionnement uniforme
    transforms.ToTensor(),                  # Conversion pixels → tenseur [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],         # Normalisation ImageNet
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 🧠 Modèles

### Modèle 1 — FaceDetectorCNN (exp_01)

CNN custom entraîné entièrement from scratch.

**Architecture :**

```
Image (224 × 224 × 3)
        ↓
Conv Block 1 : Conv2d(3→32)   + BatchNorm + ReLU + MaxPool2d   → (32, 112, 112)
        ↓
Conv Block 2 : Conv2d(32→64)  + BatchNorm + ReLU + MaxPool2d   → (64, 56, 56)
        ↓
Conv Block 3 : Conv2d(64→128) + BatchNorm + ReLU + MaxPool2d   → (128, 28, 28)
        ↓
Conv Block 4 : Conv2d(128→256)+ BatchNorm + ReLU + MaxPool2d   → (256, 14, 14)
        ↓
AdaptiveAvgPool2d(4×4)                                          → (256, 4, 4)
        ↓
Flatten                                                         → (4096,)
        ↓
Linear(4096→512) + BatchNorm + ReLU + Dropout(0.5)
        ↓
Linear(512→256)  + BatchNorm + ReLU + Dropout(0.25)
        ↓
Linear(256→1) → Logit brut → BCEWithLogitsLoss
```

**Paramètres :** 2,619,681

---

### Modèle 2 — FaceDetectorResNet18 (exp_02)

ResNet18 pré-entraîné sur ImageNet avec fine-tuning de la dernière couche.

**Architecture :**

```
Image (224 × 224 × 3)
        ↓
ResNet18 Backbone (pré-entraîné ImageNet)
        ↓
Global Average Pooling                    → (512,)
        ↓
Linear(512→256) + BatchNorm + ReLU + Dropout(0.3)
        ↓
Linear(256→1) → Logit brut → BCEWithLogitsLoss
```

**Paramètres :** 11,182,337

---

## ⚙️ Paramètres d'entraînement

| Paramètre | CNN (exp_01) | ResNet18 (exp_02) |
|-----------|:-----------:|:-----------------:|
| Batch size | 32 | 32 |
| Learning rate initial | 1e-3 | 1e-4 |
| Optimizer | Adam | Adam |
| Weight decay | 1e-4 | 1e-4 |
| Loss function | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Epochs maximum | 30 | 20 |
| Early stopping (patience) | 7 | 5 |
| Dropout | 0.5 | 0.3 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| LR patience | 3 | 3 |
| LR factor | 0.5 | 0.5 |

### Pourquoi BCEWithLogitsLoss ?

Problème de classification **binaire** (REAL=1 / FAKE=0). `BCEWithLogitsLoss` combine sigmoid + binary cross-entropy en une seule opération numériquement stable, évitant les problèmes de gradient vanishing.

### Pourquoi Adam ?

Adam adapte le learning rate par paramètre et converge rapidement sur de grands datasets. Couplé à `ReduceLROnPlateau`, il réduit automatiquement le lr quand la val_loss stagne.

---

## 📈 Résultats

### FaceDetectorCNN (exp_01)

| Époque | Train Loss | Train Acc | Val Loss | Val Acc |
|--------|-----------|-----------|----------|---------|
| 1 | 0.4699 | 76.8% | 0.3183 | 85.9% |
| 5 | 0.0807 | 97.0% | 0.0634 | 97.6% |
| 10 | 0.0436 | 98.4% | 0.1212 | 95.7% |
| 17 | 0.0147 | 99.5% | 0.0291 | 99.0% |
| 25 | 0.0059 | 99.8% | 0.0252 | 99.3% |
| **30** | **0.0044** | **99.9%** | **0.0238** | **99.1%** |

**Performance sur le test set (20 000 images) :**

| Métrique | FAKE | REAL | Global |
|----------|------|------|--------|
| Precision | 0.99 | 0.99 | 0.99 |
| Recall | 0.99 | 0.99 | 0.99 |
| F1-score | 0.99 | 0.99 | 0.99 |
| **Accuracy** | — | — | **99.31%** |
| **AUC-ROC** | — | — | **0.9999** |

---

### Comparaison des modèles

| | FaceDetectorCNN | FaceDetectorResNet18 |
|---|:---:|:---:|
| Paramètres | 2,619,681 | 11,182,337 |
| Pré-entraîné | ❌ | ✅ ImageNet |
| Accuracy test | **99.31%** | — |
| F1-score | **0.99** | — |
| AUC-ROC | **0.9999** | — |
| Epochs | 30 | 20 |
| LR initial | 1e-3 | 1e-4 |

*Les résultats ResNet18 seront mis à jour après l'entraînement.*

---

## 🚀 Installation

### 1. Clone le repo

```bash
git clone https://github.com/eya-bouhmida/ai-image-detection-cnn.git
cd ai-image-detection-cnn
```

### 2. Crée l'environnement Conda

```bash
conda create --name cnn_detection python=3.10 -y
conda activate cnn_detection
```

### 3. Installe les dépendances

```bash
pip install torch torchvision tqdm matplotlib scikit-learn seaborn gradio pillow
```

### 4. Télécharge le dataset depuis Kaggle

```bash
pip install kaggle
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip -d data/
```

---

## 🏃 Utilisation

### Entraîner le CNN from scratch

```bash
python src/train.py
```

### Entraîner ResNet18

```bash
python src/train_resnet.py
```

### Évaluer sur le test set

```bash
python -m src.evaluate
```

### Lancer le dashboard interactif

```bash
python dashboard.py
```

Puis ouvre **http://127.0.0.1:7860** dans ton navigateur.

---

### Utilisation sur Google Colab (GPU)

L'entraînement nécessite un GPU. Sur Google Colab :

```python
# 1. Clone le projet
!git clone https://github.com/eya-bouhmida/ai-image-detection-cnn.git
%cd ai-image-detection-cnn

# 2. Télécharge le dataset
!pip install kaggle -q
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces
!unzip -q 140k-real-and-fake-faces.zip -d data/

# 3. Relie les données
import os
os.makedirs('data/real_vs_fake/real-vs-fake', exist_ok=True)
!ln -sf /content/data/real_vs_fake/real-vs-fake/train data/real_vs_fake/real-vs-fake/train
!ln -sf /content/data/real_vs_fake/real-vs-fake/test data/real_vs_fake/real-vs-fake/test
!ln -sf /content/data/real_vs_fake/real-vs-fake/valid data/real_vs_fake/real-vs-fake/valid

# 4. Lance l'entraînement
!python src/train.py
```

---

## 🖥️ Dashboard Interactif

Interface développée avec **Gradio** permettant de tester le modèle en temps réel :

- 📁 Upload une image de visage (JPG, PNG, WEBP)
- 🔍 Le modèle retourne **REAL** ou **FAKE**
- 📊 Affichage de la confiance et des probabilités détaillées
- 🎨 Interface light élégante avec code couleur (vert=REAL / rouge=FAKE)

### Limitation importante

> ⚠️ Ce modèle est optimisé pour détecter les visages générés par **StyleGAN v1** uniquement.
> Les images générées par d'autres outils (MidJourney, DALL·E, StyleGAN v2/v3) peuvent ne pas être correctement détectées car le modèle a appris les artefacts spécifiques au StyleGAN utilisé dans le dataset d'entraînement.

---

## 🔬 Analyse des résultats

### Points forts
- Accuracy de **99.31%** sur 20 000 images de test
- AUC-ROC de **0.9999** — quasi parfait
- Performance symétrique sur FAKE et REAL (pas de biais de classe)
- Convergence rapide grâce au learning rate scheduler

### Limitations
- Optimisé StyleGAN v1 uniquement
- Sensible aux images de haute résolution générées par StyleGAN v2/v3
- Nécessite un GPU pour un entraînement raisonnable (100K images)

### Perspectives
- Augmenter la diversité du dataset (MidJourney, DALL·E, Stable Diffusion)
- Tester des architectures plus légères (MobileNet, EfficientNet-B0)
- Ajouter de la data augmentation (flip, rotation, color jitter)
- Déployer le dashboard sur Hugging Face Spaces

---

## 👥 Équipe

- **Eya Bouhmida**
- **Sarra Lamouchi**
- Lien github du Sarra :

---

## Demo Dashboard :
## Demo 🎥
![Demo](modèle cnn-demo/demo.gif)

