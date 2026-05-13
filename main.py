# Main script to run the application
# Generates the results and saves them for submission to Kaggle

#This script should:
#- Produce the exact submission file uploaded to Kaggle
#- Use only Python packages covered in Labs 1–3.
# ⚠️ Note: The reproducibility of your Kaggle submission is part of the grade!

from tqdm import tqdm

from utils.group_detect import *
from utils.card_separation import *
from utils.card_color_detect import *
from utils.generate_background import *
from utils.load_and_background_subtract import *
from utils.background_detection import *
from utils.token_detect import *

DATA_DIR  = "data/train_images"
TRAIN_CSV = "data/train.csv"

# --- Load the train set under the features format
print("Loading the training set...")
features = load_image_subset(DATA_DIR, subset_id=6, return_features=True)

## Step 1 — Background-type detection

# --- Persist the feature in the per-image store for downstream steps
for img_id, feat in tqdm(features.items(), desc="Detecting background type"):
    bg_type, median_sat = detect_background_type(feat["image"])
    feat["bg_type"]            = bg_type
    feat["bg_corner_median_s"] = median_sat
    
## Step 2 — Background subtraction (conditional)

# Reference background, loaded once.
REFERENCE_BG_PATH = "images/colored_background_from_crops.png"
reference_bg = load_background_image(REFERENCE_BG_PATH)

# Run on the images and persist the result.
for img_id, feat in tqdm(features.items(), desc="Preprocessing background"):
    feat["preprocessed"] = preprocess_background(feat["image"], feat["bg_type"], reference_bg)
    
## Step 3 — Card segmentation, blob extraction and player assignment

for img_id, feat in tqdm(features.items(), desc="Detecting card groups"):
    centroids, bboxes, mask = detect_groups(
        feat["image"], feat["bg_type"], preprocessed=feat["preprocessed"]
    )
    feat["mask"]            = mask
    feat["group_centroids"] = centroids
    feat["group_bboxes"]    = bboxes
    
## Step 4 — Active-player marker (token) detection

for img_id, feat in tqdm(features.items(), desc="Detecting active player"):
    active, token_xy = detect_active_player(
        feat["image"], feat["bg_type"],
        feat["group_centroids"], feat["mask"],
    )
    feat["active_player"] = active
    feat["token_xy"]      = token_xy