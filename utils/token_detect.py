import numpy as np
import cv2
from skimage.color import rgb2hsv
from skimage.morphology import dilation, disk
from skimage.measure import label, regionprops
from skimage.morphology import erosion, disk

WHITE_PARAMS = {
    "val_max":       0.35,
    "sat_max":       0.35,
    "card_erosion":  0,
    "area_min":      50,
}

NOISY_PARAMS = {
    "hue_min":       0.12,
    "hue_max":       0.16,
    "sat_min":       0.60,
    "val_min":       0.95,
    "card_erosion":  100,
    "area_min":      50,
}

def candidate_mask(image, bg_type):
    """
    Build a boolean mask of pixels matching the token's color signature
    for the given background type.

    Args:
        image (np.ndarray): RGB image, uint8 in [0, 255] or float in [0, 1].
        bg_type (str)     : 'white' or 'noisy'.

    Returns:
        np.ndarray (bool): True where the pixel matches the token signature.
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    hsv = rgb2hsv(img)

    if bg_type == "white":
        p = WHITE_PARAMS
        val, sat = hsv[:, :, 2], hsv[:, :, 1]
        return (val < p["val_max"]) & (sat < p["sat_max"])

    if bg_type == "noisy":
        p = NOISY_PARAMS
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return ((h > p["hue_min"]) & (h < p["hue_max"]) &
                (s > p["sat_min"]) & (v > p["val_min"]))

    raise ValueError(f"Unknown bg_type: {bg_type!r}.")

def erode_fast(mask, radius, downsample=4):
    """
    Erode a boolean mask with a disk of `radius` pixels — much faster than
    `skimage.morphology.erosion` for large radii by running OpenCV's
    erosion at 1/downsample resolution.

    Args:
        mask       (np.ndarray): boolean mask.
        radius     (int)       : full-resolution erosion radius.
        downsample (int)       : resolution-reduction factor.

    Returns:
        np.ndarray (bool): eroded mask, same shape as input.
    """
    if radius <= 0:
        return mask
    H, W = mask.shape
    sh, sw = H // downsample, W // downsample
    r      = max(1, radius // downsample)

    small  = cv2.resize(mask.astype(np.uint8), (sw, sh),
                        interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(small, kernel)
    return cv2.resize(eroded, (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    
def _close_fast(mask, radius):
    """
    Morphological closing of a boolean mask with an elliptical structuring
    element of `radius` pixels, using OpenCV (much faster than skimage).
    Used to heal token blobs that are fragmented (white token rim) or
    partially eroded (noisy token clipped by the card exclusion).
    """
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)

def detect_token(image, bg_type, card_mask, debug=False):
    """
    Locate the active-player token.

    Args:
        image (np.ndarray)      : original RGB image, uint8 or float.
        bg_type (str)           : 'white' or 'noisy'.
        card_mask (np.ndarray)  : cleaned card mask from `detect_groups`.
        debug (bool)            : if True, also returns intermediates and
                                  the per-blob report.

    Returns:
        Without `debug`: (cx, cy) tuple of float, or None.
        With `debug=True`: ((cx, cy) or None,
                            {'cand_raw', 'card_tight', 'cand_after',
                             'blob_report'}).
    """
    if bg_type == "white":
        p = WHITE_PARAMS
    elif bg_type == "noisy":
        p = NOISY_PARAMS
    else:
        raise ValueError(f"Unknown bg_type: {bg_type!r}.")

    cand_raw   = candidate_mask(image, bg_type)
    card_tight = erode_fast(card_mask, p["card_erosion"])
    cand_after = cand_raw & ~card_tight

    blobs = regionprops(label(cand_after, connectivity=2))
    blob_report = [(int(b.area), float(b.centroid[1]), float(b.centroid[0]))
                   for b in blobs]

    valid = [b for b in blobs if b.area >= p["area_min"]]
    if not valid:
        result = None
    else:
        best = max(valid, key=lambda b: b.area)
        cy, cx = best.centroid
        result = (float(cx), float(cy))

    if debug:
        return result, {
            "cand_raw":    cand_raw,
            "card_tight":  card_tight,
            "cand_after":  cand_after,
            "blob_report": blob_report,
        }
    return result



def detect_active_player(image, bg_type, group_centroids, card_mask):
    """
    Identify the active player from the token's pixel position.

    Calls `detect_token`, then assigns the token centroid to the player
    whose group centroid is closest in Euclidean distance. The 'center'
    group is excluded from candidates.

    Args:
        image (np.ndarray)      : original RGB image.
        bg_type (str)           : 'white' or 'noisy'.
        group_centroids (dict)  : {label: (cx, cy)} from `detect_groups`.
        card_mask (np.ndarray)  : cleaned card mask from `detect_groups`.

    Returns:
        active_player (str) or None        : 'p1'..'p4', or None if the
                                             token could not be localized.
        token_xy (tuple of float) or None  : token centroid (cx, cy) for
                                             visualization, or None.
    """
    token_xy = detect_token(image, bg_type, card_mask)
    if token_xy is None:
        return None, None

    tx, ty = token_xy
    best, best_d = None, float("inf")
    for k, (cx, cy) in group_centroids.items():
        if k == "center":
            continue
        d = (cx - tx) ** 2 + (cy - ty) ** 2
        if d < best_d:
            best_d, best = d, k

    if not best:
        return None, token_xy
    return "p" + best.split("_")[1], token_xy