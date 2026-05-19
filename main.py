# Main script to run the application
# Generates the results and saves them for submission to Kaggle

#This script should:
#- Produce the exact submission file uploaded to Kaggle
#- Use only Python packages covered in Labs 1–3.
# ⚠️ Note: The reproducibility of your Kaggle submission is part of the grade!

from tqdm import tqdm
from skimage.measure import label, regionprops
import pickle

from utils.group_detect import *
from utils.card_separation import *
from utils.card_color_detect import *
from utils.generate_background import *
from utils.load_and_background_subtract import *
from utils.background_detection import *
from utils.token_detect import *
from utils.card_separation import *
from utils.number_prediction import *
from utils.save_csv import make_submission_csv

DATA_DIR  = "data/train_images"
TRAIN_CSV = "data/train.csv"

# --- Load the train set under the features format
print("Loading the training set...")
features = load_image_subset(DATA_DIR, subset_id=7, return_features=True)

# Reference background, loaded once for step 3
REFERENCE_BG_PATH = "images/colored_background_from_crops.png"
reference_bg = load_background_image(REFERENCE_BG_PATH)

# Template and thresholds for card detection, loaded once for step 5
template = np.array(plt.imread("images/card_edge_template.png")[:,:,0], dtype=np.uint8)
THRESHOLD  = [0.9,0.8,0.75]  # multi-pass: first pass strict, second pass more permissive

# Setup for color detection (step 6)
COLOR_PREFIX = {
    "red":     "r_",
    "green":   "g_",
    "blue":    "b_",
    "yellow":  "y_",
    "special": "s_",
    "unknown": "?_",
}

for img_id, feat in tqdm(features.items(), desc="Processing images ..."):
    
    tqdm.write("--------------------------------------------------------")
    tqdm.write(f"\nProcessing image {img_id} ...")
    
    ## Step 1 — Background-type detection
    tqdm.write("Detecting background type ...")
    bg_type, median_sat = detect_background_type(feat["image"])
    feat["bg_type"]            = bg_type
    feat["bg_corner_median_s"] = median_sat
    
    ## Step 2 — Background subtraction (conditional)
    tqdm.write("Preprocessing background ...")
    feat["preprocessed"] = preprocess_background(feat["image"], feat["bg_type"], reference_bg)
    
    ## Step 3 — Card segmentation, blob extraction and player assignment
    tqdm.write("Detecting card groups ...")
    centroids, bboxes, mask = detect_groups(feat["image"], feat["bg_type"], preprocessed=feat["preprocessed"])
    feat["mask"]            = mask
    feat["group_centroids"] = centroids
    feat["group_bboxes"]    = bboxes
    
    ## Step 4 — Active-player marker (token) detection
    tqdm.write("Detecting active player ...")
    active, token_xy = detect_active_player(feat["image"], feat["bg_type"],feat["group_centroids"], feat["mask"],)
    feat["active_player"] = active
    feat["token_xy"] = token_xy
    
    ## Step 5 — Card separation within groups
    tqdm.write("Separating cards within groups ...")
    
    cleaned = feat["mask"]
    labeled_full = label(cleaned)
    _, solo, multi, tokens = classify_blobs(cleaned, SINGLE_AREA)
    rgb_img = feat["image"]
    H, W = rgb_img.shape[:2]

    all_cards  = []
    cards_list = []
    cards_with_groups = {}
    for blob in tqdm(multi + solo, desc="Processing blobs", leave=False):
        cy_b, cx_b = blob.centroid
        player = assign_label(cx_b, cy_b, W, H)
        cards, _, _, _ = detect_cards_in_blob(
            blob, labeled_full, (H, W), CARD_W, CARD_H,
            n_angles=72, img_rgb=rgb_img, template=template, threshold=THRESHOLD
        )
        all_cards.extend(cards)
        for cx, cy, ang, sw, sh, sc in cards:
            entry = (cx, cy, ang, sw, sh, sc)
            cards_list.append({
                "cx":         cx,
                "cy":         cy,
                "angle":      ang,
                "scale_w":    sw,
                "scale_h":    sh,
                "score":      sc,
                "player":     player,
                "blob_label": blob.label,
                "crop":       extract_card_crops(rgb_img, entry)
            })
            
            cards_with_groups.setdefault(player, []).append(entry)

    feat["detected_cards"] = all_cards
    feat["cards"]          = cards_list
    feat["cards_with_groups"]  = cards_with_groups

    # Step 6 — Card color detection
    img = feat["image"]
    for card in tqdm(feat["cards"], desc="Detecting card colors", leave=False):
        color = color_from_ellipse(
            img, card["cx"], card["cy"], card["angle"],
            card["scale_w"], card["scale_h"]
        )
        card["color"]        = color
        card["color_prefix"] = COLOR_PREFIX.get(color, "?_")

    feat["cards"] = [c for c in feat["cards"] if c["color"] != "unknown"]
    
    # Step 7 — Card number detection (commented for storing because takes up too much ram)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = load_number_prediction_model("src/models/bad_uno_cnn_final.pth", device)
    #_, test_transform = make_train_test_transforms()
    
    #for card in tqdm(feat["cards"], desc="Detecting card numbers", leave=False):
    #    crop = card["crop"]
    #    crop = Image.fromarray(np.array(crop, dtype=np.uint8))
    #    crop = test_transform(crop).unsqueeze(0).to(device)

    #    with torch.no_grad():
    #        output = model(crop)
    #        pred_idx = output.argmax(dim=1).item()
    #        pred_label = idx_to_label[pred_idx]

    #    card["predicted_label"] = pred_label
    #    card["full_label"] = card["color_prefix"] + pred_label


tqdm.write("---------------------------------------------------------")

# Save the extracted features for later use (e.g., for training a classifier or generating the submission file)
with open("train_features.pkl", "wb") as f:
    pickle.dump(features, f)
    
make_submission_csv(features, "submission.csv")
