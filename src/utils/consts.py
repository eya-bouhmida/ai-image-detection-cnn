import torch

# Chemins — adaptés à ta structure réelle
DATA_DIR = "data/real_vs_fake/real-vs-fake/"

TRAIN_FAKE_DIR = DATA_DIR + "train/fake/"
TRAIN_REAL_DIR = DATA_DIR + "train/real/"
TEST_FAKE_DIR  = DATA_DIR + "test/fake/"
TEST_REAL_DIR  = DATA_DIR + "test/real/"

OUTPUT_DIR = "output/"
MODEL_SAVE_PATH = OUTPUT_DIR + "best_model.pth"

# Hyperparamètres
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # real / fake

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
CLASS_NAMES = ["fake", "real"]