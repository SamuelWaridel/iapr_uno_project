"""
card_template.py
================
Build a canonical edge template for the Generalized Hough Transform (GHT)
from real UNO card images in the training set.

The template is a binary image (CARD_H × CARD_W) showing the edges of:
  1. The white rectangular frame — the border visible on every UNO card.
  2. The white oval at the card centre — the ellipse printed on every card.

These two structures appear on ALL UNO cards regardless of colour, number,
or special type, making them the most stable features for GHT-based
localisation.

Pipeline (per training image)
------------------------------
  1. Segment the image with the white-background recipe (segment_cards +
     clean_mask) to obtain binary blobs.
  2. Among all blobs, find the one whose centroid falls in the 'center'
     region AND whose area is close to one card (0.6–1.4 × SINGLE_AREA).
  3. Perspective-rectify that blob to a canonical CARD_W × CARD_H portrait
     crop using the blob's minimum-area rotated bounding rectangle.
  4. Threshold the crop in HSV to isolate white regions (frame + oval).
  5. Extract contours of the white mask and draw them on a blank canvas.
  6. Accumulate the edge image into a running sum over all images.

The accumulated image is normalised and thresholded at `threshold` (default
0.30) so that only edges that appear in at least 30% of processed images
are kept in the final template.  This removes per-card noise (numbers,
colour patches) and retains the two structures that are common to all cards.

The resulting template is saved to `images/card_edge_template.png`.

Usage
-----
    from utils.card_template import build_card_template

    template, acc = build_card_template(
        train_dir = "data/train_images",
        n_images  = 20,
        out_path  = "images/card_edge_template.png",
    )
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

from utils.group_detect import segment_cards, clean_mask, assign_label
from utils.card_separation import CARD_W, CARD_H, SINGLE_AREA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _order_corners(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the (x+y) sum to find TL/BR and the (x-y) difference to find TR/BL.
    Works for any convex quadrilateral (including slightly rotated rectangles).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left:     smallest x+y
    rect[2] = pts[np.argmax(s)]    # bottom-right: largest  x+y
    d = pts[:, 0] - pts[:, 1]     # x - y
    rect[1] = pts[np.argmax(d)]   # top-right:    largest  x-y
    rect[3] = pts[np.argmin(d)]   # bottom-left:  smallest x-y
    return rect


def _rectify_blob_to_card(img_rgb, blob, labeled):
    """
    Perspective-correct a card blob to a portrait CARD_W × CARD_H image.

    Steps:
      - Find the blob contour in the cropped bbox region (fast, small array).
      - Offset contour back to full-image coordinates.
      - Compute the minimum-area rotated bounding rectangle (minAreaRect).
      - If the rectangle is landscape (wide side on top), rotate corner order
        by one position so the output is portrait.
      - Apply getPerspectiveTransform + warpPerspective.

    Returns
    -------
    np.ndarray (CARD_H, CARD_W, 3) uint8, or None on failure.
    """
    minr, minc, maxr, maxc = blob.bbox
    blob_crop = (labeled[minr:maxr, minc:maxc] == blob.label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(blob_crop, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # Shift to full-image coordinates
    cnt_full = cnt.astype(np.int32) + np.array([[[minc, minr]]], dtype=np.int32)

    rect = cv2.minAreaRect(cnt_full)
    box  = cv2.boxPoints(rect)          # (4, 2) float32, full-image coords

    src = _order_corners(box)

    # Determine portrait vs landscape from the ordered corners
    d_top  = float(np.linalg.norm(src[1] - src[0]))   # width  of ordered rect
    d_side = float(np.linalg.norm(src[2] - src[1]))   # height of ordered rect

    if d_top > d_side:
        # Currently landscape: roll corner order by 1 → 90° CW rotation
        src = np.roll(src, 1, axis=0)

    dst = np.float32([
        [0,       0],
        [CARD_W,  0],
        [CARD_W,  CARD_H],
        [0,       CARD_H],
    ])

    try:
        M      = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img_rgb, M, (CARD_W, CARD_H))
    except cv2.error:
        return None

    return warped


def _extract_inner_contours(card_rgb, white_sat_max=0.25, white_val_min=0.70,
                             smooth_sigma=2):
    """
    Extract the two inner contours of the white regions on a rectified UNO card.

    The white mask of a UNO card is one connected region (frame + oval merged).
    When all contours are sorted by decreasing arc length:
      [0]  outer boundary of the whole white region  (= outer edge of frame)
      [1]  first inner boundary  \\  together these form the inner
      [2]  second inner boundary /  frame edge + outer oval edge

    We skip [0] and return the combination of [1] and [2], which together trace:
      - The inner edge of the white rectangular frame.
      - The outer edge of the central white oval.

    Parameters
    ----------
    card_rgb      : (CARD_H, CARD_W, 3) uint8 rectified card image.
    white_sat_max : HSV saturation upper bound for 'white'.
    white_val_min : HSV value lower bound for 'white'.
    smooth_sigma  : Gaussian sigma applied to the white mask before contouring.

    Returns
    -------
    edge_img : (CARD_H, CARD_W) uint8 binary image, 255 on edges.
    contours : list of the two kept contours (useful for the R-table builder).
    """
    img_f = card_rgb.astype(np.float32) / 255.0
    hsv   = rgb2hsv(img_f)

    raw_white = ((hsv[:, :, 1] < white_sat_max) &
                 (hsv[:, :, 2] > white_val_min)).astype(np.float32)
    smoothed  = gaussian_filter(raw_white, sigma=smooth_sigma)
    white_bin = (smoothed > 0.5).astype(np.uint8) * 255

    all_contours, _ = cv2.findContours(white_bin, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
    if len(all_contours) < 3:
        # Fallback: return all contours if fewer than 3 exist
        kept = list(all_contours)
    else:
        # Sort by arc length descending; skip [0], take [1] and [2]
        sorted_c = sorted(all_contours,
                          key=lambda c: cv2.arcLength(c, True), reverse=True)
        kept = sorted_c[1:3]

    edge_img = np.zeros((CARD_H, CARD_W), dtype=np.uint8)
    cv2.drawContours(edge_img, kept, -1, 255, thickness=2)
    return edge_img, kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_card_template(train_dir,
                         white_sat_max=0.25, white_val_min=0.70,
                         out_path="images/card_edge_template.png",
                         verbose=True):
    """
    Build and save a canonical GHT edge template from the first usable center
    card found in the training set.

    The template is derived from a single well-chosen card image so that the
    two structures are sharp and unambiguous.  Averaging over many images
    introduces misalignment blur; a single clean card gives crisper edges.

    Pipeline
    --------
    1. Iterate over white-background training images (digit -3 of stem = '7'/'8').
    2. For each image, segment blobs and pick the one closest to the image
       centre whose area matches one card (0.60–1.40 × SINGLE_AREA).
    3. Perspective-rectify the blob to CARD_W × CARD_H (portrait).
    4. Build the white mask in HSV space (low saturation, high value).
    5. Extract ALL contours with RETR_LIST; sort by arc length descending.
       - Contour [0]: outer boundary of the entire white region (outer frame).
       - Contours [1] and [2]: inner boundaries = inner frame edge + oval edge.
    6. Draw contours [1] and [2] on a blank canvas → the template.

    The template is saved to `out_path` and also returned as a uint8 array.

    Parameters
    ----------
    train_dir     : path to the directory containing training .jpg images.
    white_sat_max : HSV saturation upper bound for isolating white regions.
    white_val_min : HSV value lower bound for isolating white regions.
    out_path      : file path where the template PNG is saved (None = no save).
    verbose       : if True, print progress.

    Returns
    -------
    template : (CARD_H, CARD_W) uint8 binary image (255 = edge, 0 = bg).
    card_img : (CARD_H, CARD_W, 3) uint8 rectified card used to build it.
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    all_files = sorted(f for f in os.listdir(train_dir) if f.endswith(".jpg"))

    # Prefer white-background images (digit -3 of the stem is '7' or '8')
    white_files = [f for f in all_files
                   if len(f) >= 11 and f[:-4][-3] in ("7", "8")]
    candidates  = white_files if white_files else all_files

    if not candidates:
        raise RuntimeError(f"No .jpg images found in {train_dir}")

    for fname in candidates:
        img_path = os.path.join(train_dir, fname)
        img_rgb  = np.array(plt.imread(img_path))

        if img_rgb.dtype != np.uint8:
            img_rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
        if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]

        H, W = img_rgb.shape[:2]

        # ── Segment and find center blob ──────────────────────────────────
        mask        = segment_cards(img_rgb)
        cleaned     = clean_mask(mask, close_radius=60, open_radius=5,
                                 min_blob_size=5000)
        labeled_img = label(cleaned)
        blobs       = regionprops(labeled_img)

        center_blob = None
        best_dist   = float("inf")
        for b in blobs:
            cy_b, cx_b = b.centroid
            if not (0.60 < b.area / SINGLE_AREA < 1.40):
                continue
            if assign_label(cx_b, cy_b, W, H) != "center":
                continue
            dist = np.hypot(cx_b - W / 2, cy_b - H / 2)
            if dist < best_dist:
                best_dist, center_blob = dist, b

        if center_blob is None:
            if verbose:
                print(f"  [{fname}] no center single-card blob — skipped")
            continue

        # ── Rectify ───────────────────────────────────────────────────────
        card_img = _rectify_blob_to_card(img_rgb, center_blob, labeled_img)
        if card_img is None:
            if verbose:
                print(f"  [{fname}] rectification failed — skipped")
            continue

        # ── Extract inner contours [1] and [2] ────────────────────────────
        template, _ = _extract_inner_contours(card_img, white_sat_max,
                                              white_val_min)
        edge_px = int((template > 0).sum())

        if edge_px < 500:
            if verbose:
                print(f"  [{fname}] too few edge pixels ({edge_px}) — skipped")
            continue

        if verbose:
            area_pct = center_blob.area / SINGLE_AREA * 100
            print(f"  [{fname}] OK  area={area_pct:.0f}%  edge_px={edge_px}")
            print(f"Template built from {fname}")

        # ── Save ──────────────────────────────────────────────────────────
        if out_path is not None:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plt.imsave(out_path, template, cmap="gray")
            if verbose:
                print(f"Template saved → {out_path}")

        return template, card_img

    raise RuntimeError(
        "Could not extract a valid center card from any training image.\n"
        "Check 'train_dir', SINGLE_AREA, CARD_W, CARD_H."
    )


def visualize_template(template, acc=None, title="Card edge template"):
    """
    Display the template (and optionally the raw accumulator) with matplotlib.

    Parameters
    ----------
    template : (CARD_H, CARD_W) uint8 binary template from build_card_template.
    acc      : (CARD_H, CARD_W) float32 accumulator (optional, shown if provided).
    title    : figure title.
    """
    n_plots = 2 if acc is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    axes[0].imshow(template, cmap="gray")
    axes[0].set_title("Binary template (threshold applied)")
    axes[0].axis("off")

    if acc is not None:
        axes[1].imshow(acc, cmap="hot")
        axes[1].set_title("Raw accumulator\n(brighter = more images agree)")
        axes[1].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
