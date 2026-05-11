import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from collections import defaultdict
from PIL import Image
import cv2


def load_image(image_path):
    """
    Load an image from disk and return it as a NumPy RGB array.
 
    Args:
        image_path (str): path to the image file.
 
    Returns:
        img (np.ndarray): RGB image of shape (H, W, 3), dtype uint8.
    """
    img = np.array(Image.open(image_path))
    return img

def segment_cards(img):
    """
    Produce a binary mask that isolates UNO cards from the background.
 
    Strategy: the table background is a uniform light gray
    (high brightness, very low saturation). Cards are either white-bordered
    or brightly colored — both differ from the background in saturation.
 
    We keep pixels that are:
      - bright enough      (value  > 0.75)
      - NOT the gray background  (saturation < 0.08 AND value > 0.80)
 
    Args:
        img (np.ndarray): RGB image, shape (H, W, 3), values in [0, 255].
 
    Returns:
        mask (np.ndarray): boolean mask, True where cards are present.
    """
    # Convert to HSV for easier color-based thresholding
    img_norm = img.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm = img_norm / 255.0
    hsv = rgb2hsv(img_norm)
    val = hsv[:, :, 2]   # brightness channel
    sat = hsv[:, :, 1]   # saturation channel
 
    # Keep bright pixels that are not the flat gray background
    mask = (val > 0.75) & ~((sat < 0.08) & (val > 0.80))
    return mask

def clean_mask(mask, close_radius=60, open_radius=5, min_blob_size=5000,
               downsample=4, pre_open_radius=0, pre_close_min_blob_size=0): 
    H, W = mask.shape
    sh, sw = H // downsample, W // downsample
    small = cv2.resize(mask.astype(np.uint8) * 255,
                       (sw, sh), interpolation=cv2.INTER_NEAREST)

    r_pre   = max(0, pre_open_radius // downsample)    
    r_close = max(1, close_radius    // downsample)
    r_open  = max(1, open_radius     // downsample)

    if r_pre > 0:                                      
        k_pre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2*r_pre+1, 2*r_pre+1))
        small = cv2.morphologyEx(small, cv2.MORPH_OPEN, k_pre)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_close+1, 2*r_close+1))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r_open+1,  2*r_open+1))
    
    small_cleaned = remove_small_objects(small > 0, min_size=pre_close_min_blob_size).astype(np.uint8) * 255

    closed = cv2.morphologyEx(small_cleaned,  cv2.MORPH_CLOSE, k_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  k_open)
    full   = cv2.resize(opened, (W, H), interpolation=cv2.INTER_NEAREST)
    return remove_small_objects(full > 0, min_size=min_blob_size)

def extract_blobs(cleaned_mask):
    """
    Label connected components in the cleaned mask and return their properties.
 
    Args:
        cleaned_mask (np.ndarray): boolean mask from clean_mask().
 
    Returns:
        blobs        (list)      : list of RegionProperties objects (one per blob).
        centroids_xy (np.ndarray): array of shape (N, 2) with (cx, cy) per blob.
        areas        (np.ndarray): array of shape (N,) with pixel area per blob.
    """
    labeled = label(cleaned_mask)
    blobs   = regionprops(labeled)
 
    # regionprops returns centroid as (row, col) = (cy, cx), so we swap
    centroids_xy = np.array([[r.centroid[1], r.centroid[0]] for r in blobs])
    areas        = np.array([r.area for r in blobs])
 
    return blobs, centroids_xy, areas

# in utils/group_detect.py — alongside merge_blobs_by_label

def filter_blobs_by_appearance(blobs, centroids_xy, areas, image,
                               white_ratio_min=0.05):
    """
    Discard blobs that do not look like UNO cards in the original image.

    The cleaned mask produced for noisy-background images can contain
    card-sized false positives created by residuals of the leaf pattern
    where the reference background does not perfectly match the input
    (e.g. slight camera-position drift between the reference shots and
    the test image). These false positives survive size-based filtering
    because they are roughly card-shaped and card-sized; what they lack
    is the *appearance* of an UNO card.

    Real UNO cards have a thick white border, which gives them a
    measurable fraction of pixels that are simultaneously very bright
    and very low-saturation in the original RGB image
    (`val > 0.85 AND sat < 0.15`). Leaf-pattern blobs have virtually
    none. We use this single, robust feature to keep only "card-like"
    blobs.

    Args:
        blobs        (list)      : RegionProperties returned by
                                   `extract_blobs()` on the cleaned mask.
        centroids_xy (np.ndarray): (N, 2) centroid coordinates from
                                   `extract_blobs()`.
        areas        (np.ndarray): (N,) blob areas from `extract_blobs()`.
        image        (np.ndarray): original RGB image (uint8 in [0, 255]
                                   or float in [0, 1]). Sampled inside
                                   each blob's pixel set, NOT the
                                   bounding box, so neighbouring leaves
                                   do not bias the score.
        white_ratio_min (float)  : minimum fraction of white-border-like
                                   pixels (default 0.05) for a blob to
                                   be kept. Lower values are more
                                   permissive.

    Returns:
        kept_blobs        (list)       : surviving blobs (subset of input).
        kept_centroids_xy (np.ndarray) : matching centroids,
                                         shape (K, 2) with K <= N.
        kept_areas        (np.ndarray) : matching areas, shape (K,).
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    keep = []
    for i, blob in enumerate(blobs):
        coords  = blob.coords                    # (Npx, 2) (row, col)
        pixels  = img[coords[:, 0], coords[:, 1]]  # (Npx, 3) RGB
        hsv     = rgb2hsv(pixels[None, :, :])[0]   # (Npx, 3) HSV
        val, sat = hsv[:, 2], hsv[:, 1]
        white_ratio = float(((val > 0.85) & (sat < 0.15)).mean())
        if white_ratio >= white_ratio_min:
            keep.append(i)

    if not keep:
        return [], np.empty((0, 2)), np.empty(0)

    keep = np.array(keep, dtype=int)
    return ([blobs[i] for i in keep],
            centroids_xy[keep],
            areas[keep])

def merge_blobs(blobs, centroids_xy, areas, merge_dist_ratio=0.20, img_width=None, img_height=None):
    """
    Group blobs whose centroids are closer than a distance threshold.
 
    Cards of the same player that were not fully connected after morphology
    will have nearby centroids. We merge them into one group using a
    Union-Find (disjoint set) structure.
 
    The merge threshold is expressed as a fraction of the image width so
    that it scales automatically with image resolution.
 
    Args:
        blobs            (list)      : RegionProperties list from extract_blobs().
        centroids_xy     (np.ndarray): (N, 2) centroid coordinates.
        areas            (np.ndarray): (N,) blob areas used as weights.
        merge_dist_ratio (float)     : merge threshold as a fraction of image
                                       width (default 0.20 = 20%).
        img_width        (int)       : image width in pixels.
 
    Returns:
        group_centroids (dict): {label_str: (cx, cy)}  weighted centroid per group.
        group_bboxes    (dict): {label_str: (minr, minc, maxr, maxc)} bounding box.
    """
    MERGE_DIST = img_width * merge_dist_ratio
 
    # --- Union-Find helpers ---
    parent = list(range(len(blobs)))
 
    def find(i):
        # Path compression: flatten the tree on the way up
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
 
    def union(i, j):
        parent[find(i)] = find(j)
 
    # Merge blobs whose centroids are within MERGE_DIST of each other
    dists = cdist(centroids_xy, centroids_xy)
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            if dists[i, j] < MERGE_DIST:
                union(i, j)
 
    # Collect blob indices per cluster
    clusters = defaultdict(list)
    for i in range(len(blobs)):
        clusters[find(i)].append(i)
 
    # Compute weighted centroid and enclosing bounding box for each cluster
    group_centroids = {}
    group_bboxes    = {}
 
    for idxs in clusters.values():
        w  = areas[idxs]
 
        # Area-weighted centroid gives more importance to larger card regions
        cx = np.average(centroids_xy[idxs, 0], weights=w)
        cy = np.average(centroids_xy[idxs, 1], weights=w)
 
        # Assign a semantic label based on position in the image
        lbl = assign_label(cx, cy, img_width, img_height)
 
        # Enclosing bbox: smallest box that covers all blobs in the cluster
        minr = min(blobs[i].bbox[0] for i in idxs)
        minc = min(blobs[i].bbox[1] for i in idxs)
        maxr = max(blobs[i].bbox[2] for i in idxs)
        maxc = max(blobs[i].bbox[3] for i in idxs)
 
        group_centroids[lbl] = (cx, cy)
        group_bboxes[lbl]    = (minr, minc, maxr, maxc)
 
    return group_centroids, group_bboxes

def merge_blobs_by_label(blobs, centroids_xy, areas, img_width, img_height):
    """
    Group blobs by the semantic label of their centroid (player_1..4 / center).
    Unlike merge_blobs, this can never merge blobs across different player
    regions, which makes it robust to fragmented masks where small sub-blobs
    of different players might lie within a few hundred pixels of each other.
    """
    clusters = defaultdict(list)
    for i, (cx, cy) in enumerate(centroids_xy):
        lbl = assign_label(cx, cy, img_width, img_height)
        clusters[lbl].append(i)

    group_centroids, group_bboxes = {}, {}
    for lbl, idxs in clusters.items():
        w = areas[idxs]
        cx = np.average(centroids_xy[idxs, 0], weights=w)
        cy = np.average(centroids_xy[idxs, 1], weights=w)
        minr = min(blobs[i].bbox[0] for i in idxs)
        minc = min(blobs[i].bbox[1] for i in idxs)
        maxr = max(blobs[i].bbox[2] for i in idxs)
        maxc = max(blobs[i].bbox[3] for i in idxs)
        group_centroids[lbl] = (cx, cy)
        group_bboxes[lbl]    = (minr, minc, maxr, maxc)
    return group_centroids, group_bboxes

def assign_label(cx, cy, img_width, img_height=None):
    """
    Map a centroid (cx, cy) to one of five semantic labels based on its
    position in the image.

    Layout convention (top-down photo of an UNO game):
      - player_1 : bottom (rel_y > 0.65)
      - player_2 : right  (rel_x > 0.65)
      - player_3 : top    (rel_y < 0.35)
      - player_4 : left   (rel_x < 0.35)
      - center   : middle (0.30 < rel_x < 0.70 AND 0.30 < rel_y < 0.70)

    Args:
        cx, cy            : centroid coordinates in pixels.
        img_width         : image width  in pixels.
        img_height        : image height in pixels (estimated from cy if None).

    Returns:
        str : one of 'player_1'..'player_4', 'center'.
    """
    H     = img_height if img_height is not None else cy * 2
    rel_x = cx / img_width
    rel_y = cy / H

    if 0.30 < rel_x < 0.70 and 0.30 < rel_y < 0.70:
        return "center"
    if rel_y > 0.65:
        return "player_1"
    if rel_y < 0.35:
        return "player_3"
    if rel_x < 0.35:
        return "player_4"
    return "player_2"


def crop_regions(img, group_bboxes):
    """
    Extract image crops for each detected group using their bounding boxes.
 
    Args:
        img          (np.ndarray): original RGB image, shape (H, W, 3).
        group_bboxes (dict)      : {label: (minr, minc, maxr, maxc)}.
 
    Returns:
        regions (dict): {label: np.ndarray crop of shape (h, w, 3)}.
    """
    regions = {}
    for lbl, (minr, minc, maxr, maxc) in group_bboxes.items():
        regions[lbl] = img[minr:maxr, minc:maxc]
    return regions

def segment_cards_noisy_background(img, val_quantile=0.90, sat_quantile=0.05, plot=False):
    if img.max() > 1.0:
        img = img / 255.0
    hsv = rgb2hsv(img)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    saturation_threshold = np.quantile(sat, sat_quantile)
    value_threshold      = np.quantile(val, val_quantile)
    mask = (sat > saturation_threshold) & (val > value_threshold)
    # (plot block unchanged)
    return mask

# Per-background-type cleaning recipes — single source of truth.
CLEAN_PARAMS = {
    "white": dict(close_radius=60, open_radius=5,  min_blob_size=5000, pre_open_radius=0),
    "noisy": dict(close_radius=30, open_radius=2,  min_blob_size=5000, pre_open_radius=24),
}


PIPELINE_PARAMS = {
    "white": {
        "seg_kwargs":   {},
        "clean_kwargs": dict(close_radius=60, open_radius=5, min_blob_size=5000,
                             pre_open_radius=0),
    },
    "noisy": {
        "seg_kwargs":   dict(val_quantile=0.90, sat_quantile=0.05),
        "clean_kwargs": dict(close_radius=80, open_radius=2, min_blob_size=5000,
                            pre_open_radius=4, pre_close_min_blob_size=50),
    },
}

def detect_groups(image, bg_type, preprocessed=None):
    """
    Step 3 of the pipeline: turn an image (and its background-subtracted
    version, if any) into a `{label: bbox}` dictionary covering the five
    regions of interest (`player_1..4`, `center`).

    Dispatches on `bg_type` and applies the segmentation/cleaning recipe
    in `PIPELINE_PARAMS`. For the noisy path, a post-segmentation
    appearance filter (`filter_blobs_by_appearance`) discards card-sized
    blobs that lack the white-border signature of an UNO card —
    typically caused by residuals of the leaf pattern when the reference
    background is slightly misaligned with the input. The white path
    skips this filter because its segmentation already excludes the
    grey table by saturation alone.

    Connected components are then grouped by spatial label using
    `merge_blobs_by_label`, which assigns each blob to one of
    `player_1..4` or `center` and unions the bounding boxes per label.

    Args:
        image (np.ndarray)        : original RGB image, uint8 in [0, 255].
                                    Used directly for the white path and
                                    as the appearance reference for the
                                    noisy filter.
        bg_type (str)             : 'white' or 'noisy', as returned by
                                    `detect_background_type`.
        preprocessed (np.ndarray) : background-subtracted image, float in
                                    [0, 1]. Required when bg_type ==
                                    'noisy'; ignored for the white path.

    Returns:
        group_centroids (dict)        : {label: (cx, cy)} area-weighted
                                        centroid of each detected region.
        group_bboxes    (dict)        : {label: (minr, minc, maxr, maxc)}
                                        bounding box of each detected region.
        cleaned_mask    (np.ndarray)  : boolean mask after morphology and
                                        small-object filtering. Kept around
                                        because step 5 (GHT) consumes it.
    """
    p = PIPELINE_PARAMS[bg_type]

    if bg_type == "white":
        mask = segment_cards(image)
    elif bg_type == "noisy":
        if preprocessed is None:
            raise ValueError("`preprocessed` is required for noisy backgrounds.")
        mask = segment_cards_noisy_background(preprocessed, **p["seg_kwargs"])
    else:
        raise ValueError(f"Unknown bg_type: {bg_type!r}. "
                         "Expected 'white' or 'noisy'.")

    cleaned = clean_mask(mask, **p["clean_kwargs"])

    blobs, centroids_xy, areas = extract_blobs(cleaned)
    if len(blobs) == 0:
        return {}, {}, cleaned

    if bg_type == "noisy":
        blobs, centroids_xy, areas = filter_blobs_by_appearance(
            blobs, centroids_xy, areas, image, white_ratio_min=0.05
        )
        if len(blobs) == 0:
            return {}, {}, cleaned

    H, W = cleaned.shape
    group_centroids, group_bboxes = merge_blobs_by_label(
        blobs, centroids_xy, areas, img_width=W, img_height=H,
    )
    return group_centroids, group_bboxes, cleaned