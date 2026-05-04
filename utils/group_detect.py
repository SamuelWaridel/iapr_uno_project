import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from collections import defaultdict
from PIL import Image


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
    hsv = rgb2hsv(img / 255.0)
    val = hsv[:, :, 2]   # brightness channel
    sat = hsv[:, :, 1]   # saturation channel
 
    # Keep bright pixels that are not the flat gray background
    mask = (val > 0.75) & ~((sat < 0.08) & (val > 0.80))
    return mask

def clean_mask(mask, close_radius=60, open_radius=5, min_blob_size=5000):
    """
    Apply morphological closing and opening to the binary mask, then remove
    small spurious blobs.
 
    - Closing (large disk): bridges the small gaps between adjacent cards
      in the same hand, merging them into a single connected region.
    - Opening (small disk): removes thin noise and isolated bright pixels
      that survived the threshold step.
    - remove_small_objects: discards any remaining tiny blobs that are
      clearly not cards (dust, reflections, etc.).
 
    Args:
        mask          (np.ndarray): boolean mask from segment_cards().
        close_radius  (int)       : radius of the closing disk (default 60 px).
        open_radius   (int)       : radius of the opening disk (default 5 px).
        min_blob_size (int)       : minimum blob area to keep, in pixels.
 
    Returns:
        cleaned (np.ndarray): cleaned boolean mask.
    """
    # Close gaps between cards of the same hand
    closed = closing(mask, disk(close_radius))
 
    # Remove thin noise introduced or left by closing
    opened = opening(closed, disk(open_radius))
 
    # Discard blobs that are too small to be a card
    cleaned = remove_small_objects(opened, min_size=min_blob_size)
    return cleaned

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

def merge_blobs(blobs, centroids_xy, areas, merge_dist_ratio=0.20, img_width=None):
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
        lbl = assign_label(cx, cy, img_width)
 
        # Enclosing bbox: smallest box that covers all blobs in the cluster
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
    relative position in the image.
 
    Layout convention (standard UNO top-down photo):
      - player_1 : top    (rel_y < 0.35)
      - player_2 : left   (rel_x < 0.35)
      - player_3 : bottom (rel_y > 0.65)
      - player_4 : right  (rel_x > 0.65)
      - center   : middle (0.30 < rel_x < 0.70 AND 0.30 < rel_y < 0.70)
 
    Args:
        cx         (float): centroid x coordinate in pixels.
        cy         (float): centroid y coordinate in pixels.
        img_width  (int)  : image width in pixels.
        img_height (int)  : image height in pixels (estimated from cy if None).
 
    Returns:
        label (str): one of 'player_1', 'player_2', 'player_3', 'player_4', 'center'.
    """
    H      = img_height if img_height is not None else cy * 2
    rel_x  = cx / img_width
    rel_y  = cy / H
 
    if 0.30 < rel_x < 0.70 and 0.30 < rel_y < 0.70:
        return "center"
    if rel_y < 0.35:
        return "player_1"
    if rel_y > 0.65:
        return "player_3"
    if rel_x < 0.35:
        return "player_2"
    return "player_4"


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


