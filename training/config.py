from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATASET_DIR = BASE_DIR / "dataset" / "brain_mri"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

# Model
MODEL_DIR = BASE_DIR / "backend" / "trained_model"
MODEL_PATH = MODEL_DIR / "model.keras"

# Training
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]