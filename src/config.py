"""
config.py — All paths and hyper-parameters in one place.
=========================================================
Paper: UCDNet (Basavaraju et al., IEEE TGRS 2022)

Changes from previous version:
  - Removed FOCAL_GAMMA, TVERSKY_ALPHA, TVERSKY_BETA  (not in paper)
  - Removed OVERSAMPLE_RATIO                           (not in paper)
  - Removed NO_CHANGE_RATIO                            (not in paper)
  - Removed TRAIN_CITIES / VAL_CITIES / TEST_CITIES    (paper splits by patch, not city)
  - EPOCHS set to 30                                   (paper: 30 epochs)
  - CLASS_WEIGHTS set to (0.1, 0.9)                    (paper default)
  - BATCH_SIZE = 1                                     (paper: batch size 1)
  - LEARNING_RATE = 0.0001                             (paper: Adam lr=0.0001)
"""

import os

# ── 1. DATASET PATHS ───────────────────────────────────────────────────────
# Set via environment variable or change the fallback path below
DATASET_ROOT = os.environ.get(
    "OSCD_DATASET_ROOT",
    r"D:\UCDNET\onera-satellite-change-detection-dataset"   # ← CHANGE THIS (or set env var)
)

IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "train_labels")

# ── 2. ALL 24 CITIES ───────────────────────────────────────────────────────
# Paper: patches extracted from ALL 24 cities, then randomly split 200/57/30
ALL_CITIES = [
    "aguasclaras", "bercy", "bordeaux",
    "nantes", "paris", "rennes", "saclay_e",
    "abudhabi", "cupertino", "pisa", "beihai",
    "hongkong", "beirut", "mumbai", "brasilia",
    "montpellier", "norcia", "rio", "saclay_w",
    "valencia", "dubai", "lasvegas", "milano",
    "chongqing",
]

# ── 3. PATCH SPLIT COUNTS (paper: 200 train / 57 val / 30 test) ───────────
N_TRAIN_PATCHES = 200
N_VAL_PATCHES   = 57
N_TEST_PATCHES  = 30

# ── 4. MODEL / TRAINING HYPER-PARAMETERS ──────────────────────────────────
INPUT_SHAPE   = (512, 512, 13)   # 512×512 patches, 13 Sentinel-2 bands
NUM_CLASSES   = 2                # changed / unchanged
BATCH_SIZE    = 1                # paper: batch_size = 1
EPOCHS        = 60               # paper: 30 epochs  (set higher; early-stop will halt)
LEARNING_RATE = 0.0001           # paper: Adam lr = 0.0001

# ── 5. LOSS FUNCTION WEIGHTS ───────────────────────────────────────────────
# Paper Eq. 14: WCCE class weights (unchanged=0.1, changed=0.9)
CLASS_WEIGHTS = (0.1, 0.9)

# ── 6. OUTPUT PATHS ────────────────────────────────────────────────────────
OUTPUT_DIR      = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")
CURVES_PATH     = os.path.join(OUTPUT_DIR, "training_curves.png")
METRICS_PATH    = os.path.join(OUTPUT_DIR, "metrics.csv")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(OUTPUT_DIR,      exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)