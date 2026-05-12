import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from collections import defaultdict
from PIL import Image
import cv2

CARD_W      = 353
CARD_H      = 576
SINGLE_AREA = CARD_W * CARD_H 

def classify_blobs(filled, single_area, overlap_ratio=1.3,token_ratio=0.7):
    """
    Label connected components and classify each blob as containing
    one card (solo) or multiple overlapping cards (multi).
 
    A blob with area > overlap_ratio * single_area is assumed to
    contain at least two overlapping cards.
 
    Args:
        filled        (np.ndarray): filled boolean mask from clean_mask().
        single_area   (float)     : reference area of one card in pixels.
        overlap_ratio (float)     : area threshold to flag as multi-card.
 
    Returns:
        labeled     (np.ndarray): integer label image.
        solo_blobs  (list)      : RegionProperties of single-card blobs.
        multi_blobs (list)      : RegionProperties of multi-card blobs.
    """
    labeled     = label(filled)
    blobs       = regionprops(labeled)
    solo_blobs  = [b for b in blobs if b.area <= overlap_ratio * single_area and b.area >= token_ratio * single_area]
    token_blobs  = [b for b in blobs if b.area < token_ratio * single_area]
    multi_blobs = [b for b in blobs if b.area >  overlap_ratio * single_area]
    return labeled, solo_blobs, multi_blobs, token_blobs

def get_blob_contour(blob, labeled, img_shape, margin_ratio=0.4):
    """
    Extract only the outer contour of a blob as a binary image.

    Using RETR_EXTERNAL ensures we capture only the blob boundary
    and not any internal structure (card numbers, oval pattern, etc.).
    This clean signal is what the GHT will vote against.

    Args:
        blob         : RegionProperties of the blob.
        labeled      : integer label image from classify_blobs().
        img_shape    : (H, W) of the full image.
        margin_ratio : fraction of max(card_w, card_h) to add as padding.

    Returns:
        contour_img (np.ndarray): uint8 image, 255 on contour, 0 elsewhere.
        offset      (tuple)     : (off_x, off_y) top-left in full image coords.
        blob_mask   (np.ndarray): uint8 filled blob (255 inside, 0 outside).
                                  Used by ght_accumulate for gradient computation.
    """
    margin = int(max(CARD_W, CARD_H) * margin_ratio)
    minr, minc, maxr, maxc = blob.bbox
    r0 = max(0, minr - margin);  r1 = min(img_shape[0], maxr + margin)
    c0 = max(0, minc - margin);  c1 = min(img_shape[1], maxc + margin)

    blob_mask = (labeled[r0:r1, c0:c1] == blob.label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros_like(blob_mask)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=2)

    # Fill the outer contour locally (fast on small crop) so that Sobel
    # gradients in ght_accumulate point cleanly inward without hole noise.
    blob_mask_filled = np.zeros_like(blob_mask)
    cv2.drawContours(blob_mask_filled, contours, -1, 255, thickness=cv2.FILLED)
    return contour_img, (c0, r0), blob_mask_filled

def build_r_vectors(card_w, card_h, angle_deg, n_pts=80):
    """
    Compute the R-table entry for one card rotation angle.
 
    The R-table is the core of the Generalized Hough Transform.
    For each point on the card perimeter, it stores the vector
    pointing from that point back to the card center.
 
    During detection, every contour pixel casts votes by adding
    these vectors to its own position — where votes accumulate
    is where the card center must be.
 
    This function is fully vectorized with NumPy (no Python loops
    over points), which makes it fast enough for real-time use.
 
    Args:
        card_w, card_h (int)  : card dimensions in pixels.
                                CARD_W is always the short side,
                                CARD_H the long side (portrait).
                                Horizontal cards are handled by the
                                angle range covering 0°→±90°.
        angle_deg      (float): card rotation angle in degrees.
                                0° = portrait, ±90° ≈ landscape.
        n_pts          (int)  : total sample points along 4 sides.

    Returns:
        r_vecs      (np.ndarray): shape (n_pts, 2), dtype float32.
                                  Each row (dx, dy) is the vector from
                                  a perimeter sample to the card center.
        grad_angles (np.ndarray): shape (n_pts,), dtype float32.
                                  Inward normal angle (radians) at each
                                  perimeter sample. Matches the Sobel
                                  gradient direction on a bright blob:
                                  top-side → ~+90°, right → ~180°, etc.
    """
    a  = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot    = np.array([[ca, -sa], [sa, ca]])
    hw, hh = card_w / 2, card_h / 2

    # 4 corners of the card in local frame (center = origin)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    corners = (rot @ corners.T).T   # rotate to global frame

    # Inward normals per side (local frame, before card rotation).
    # These match Sobel on a bright blob on dark background:
    #   top side  → gradient points down  = (0, +1)
    #   right side→ gradient points left  = (-1, 0)
    #   bottom    → gradient points up    = (0, -1)
    #   left side → gradient points right = (+1, 0)
    inward_normals = np.array([
        [ 0,  1],
        [-1,  0],
        [ 0, -1],
        [ 1,  0],
    ], dtype=float)

    pts_per_side = n_pts // 4
    perim_pts   = []
    grad_angles = []

    for i in range(4):
        p0, p1 = corners[i], corners[(i+1) % 4]
        ts  = np.linspace(0, 1, pts_per_side, endpoint=False)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        perim_pts.append(pts)

        # Rotate the inward normal by the card angle
        n_rot = rot @ inward_normals[i]
        grad_angles.extend([np.arctan2(n_rot[1], n_rot[0])] * pts_per_side)

    perim_pts   = np.vstack(perim_pts)
    r_vecs      = -perim_pts.astype(np.float32)
    grad_angles = np.array(grad_angles, dtype=np.float32)
    return r_vecs, grad_angles

def ght_accumulate(contour_img, card_w, card_h, angles_deg, n_pts=80,
                   blob_mask=None, n_angle_bins=8, scales=None):
    """
    Accumulate votes in 2D Hough space using the R-table with
    gradient-direction filtering and anisotropic multi-scale support.

    Gradient filtering: each contour pixel votes only with the r_vecs
    whose gradient angle matches its own (±1 bin + opposite 180° bin).
    This eliminates ~half of spurious votes and sharpens the peaks.

    Multi-scale: the inner loop runs over every (scale, angle) pair so
    that cards closer or farther from the camera still produce strong peaks
    even if their true size differs from the nominal card_w × card_h.

    Anisotropic scaling: scales can contain (sw, sh) tuples to test
    independent width/height multipliers, which helps distinguish the
    second card in overlapping pairs from false 90°-rotated half-card peaks.

    Args:
        contour_img   (np.ndarray): uint8 outer contour, shape (H, W).
        card_w/h      (int)       : nominal card dimensions.
        angles_deg    (list)      : angles to test in degrees.
        n_pts         (int)       : perimeter sample points per angle.
        blob_mask     (np.ndarray): uint8 filled blob mask (same shape).
                                    Used for clean Sobel gradients.
                                    Falls back to contour_img if None.
        n_angle_bins  (int)       : gradient direction bins (default 8 = 45°/bin).
        scales        (list)      : scale multipliers to test. Each entry can be:
                                    - float s   → isotropic (s, s)
                                    - (sw, sh)  → anisotropic width/height scales
                                    Defaults to [(1.0, 1.0)] (fixed size).

    Returns:
        acc_norm        (np.ndarray): normalized accumulator [0-1], shape (H, W).
        best_angle_map  (np.ndarray): float32, best card angle (degrees) per pixel.
        best_scale_w_map(np.ndarray): float32, best width scale per pixel.
        best_scale_h_map(np.ndarray): float32, best height scale per pixel.
    """
    if scales is None:
        scales = [1.0]

    # Normalize: convert scalar scales to (sw, sh) tuples
    norm_scales = [(s, s) if isinstance(s, (int, float)) else tuple(s) for s in scales]

    rh, rw = contour_img.shape
    acc              = np.zeros((rh, rw), dtype=np.float32)
    best_angle_map   = np.zeros((rh, rw), dtype=np.float32)
    best_scale_w_map = np.ones((rh, rw),  dtype=np.float32)
    best_scale_h_map = np.ones((rh, rw),  dtype=np.float32)

    ys, xs = np.where(contour_img > 0)
    if len(xs) == 0:
        return acc, best_angle_map, best_scale_w_map, best_scale_h_map

    # Gradient angles at contour pixels — computed from the filled blob mask
    # (bright region on black) for clean inward-pointing normals.
    grad_src = blob_mask.astype(np.float32) if blob_mask is not None \
               else contour_img.astype(np.float32)
    gx = cv2.Sobel(grad_src, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(grad_src, cv2.CV_32F, 0, 1, ksize=5)
    pixel_grad = np.arctan2(gy, gx)[ys, xs]   # (N,) in [-π, π]

    pixel_coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

    bin_size = 2 * np.pi / n_angle_bins
    p_bins   = np.floor((pixel_grad + np.pi) / bin_size).astype(int) % n_angle_bins
    half     = n_angle_bins // 2

    for sw, sh in norm_scales:
        w = int(card_w * sw)
        h = int(card_h * sh)

        for angle_deg in angles_deg:
            r_vecs, r_grad_angles = build_r_vectors(w, h, angle_deg, n_pts)
            r_bins = np.floor((r_grad_angles + np.pi) / bin_size).astype(int) % n_angle_bins

            acc_angle = np.zeros((rh, rw), dtype=np.float32)

            for b in range(n_angle_bins):
                p_mask = (p_bins == b)
                # ±1 bin around b AND around the opposite bin (180° ambiguity)
                opp    = (b + half) % n_angle_bins
                r_mask = ((r_bins == b)                        |
                          (r_bins == (b   - 1) % n_angle_bins) |
                          (r_bins == (b   + 1) % n_angle_bins) |
                          (r_bins == opp)                      |
                          (r_bins == (opp - 1) % n_angle_bins) |
                          (r_bins == (opp + 1) % n_angle_bins))

                if not p_mask.any() or not r_mask.any():
                    continue

                px_b = pixel_coords[p_mask]   # (N_b, 2)
                rv_b = r_vecs[r_mask]         # (M_b, 2)

                centers = px_b[:, None, :] + rv_b[None, :, :]
                cx_all  = centers[:, :, 0].ravel().astype(np.int32)
                cy_all  = centers[:, :, 1].ravel().astype(np.int32)
                valid   = (cx_all >= 0) & (cx_all < rw) & (cy_all >= 0) & (cy_all < rh)
                np.add.at(acc_angle, (cy_all[valid], cx_all[valid]), 1.0)

            improved = acc_angle > acc
            acc[improved]              = acc_angle[improved]
            best_angle_map[improved]   = angle_deg
            best_scale_w_map[improved] = sw
            best_scale_h_map[improved] = sh

    acc = cv2.GaussianBlur(acc, (21, 21), 0)
    acc_norm = acc / (acc.max() + 1e-8)
    return acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map


def find_peaks(acc_norm, best_angle_map, n_cards, card_w, card_h,
               threshold=0.25, best_scale_w_map=None, best_scale_h_map=None):
    """
    Extract card center positions from the accumulator via greedy
    non-maximum suppression.

    Algorithm:
      1. Find global max → card center candidate.
      2. If score < threshold, stop.
      3. Suppress a neighborhood of radius min(card_w,card_h)*0.4
         around the peak to avoid selecting the same card twice.
      4. Repeat until n_cards found or score drops below threshold.

    Args:
        acc_norm         (np.ndarray): normalized accumulator from ght_accumulate().
        best_angle_map   (np.ndarray): angle at max vote, from ght_accumulate().
        n_cards          (int)       : expected number of cards in this blob.
        card_w/h         (int)       : nominal card dimensions (suppression radius).
        threshold        (float)     : min score to accept a detection [0-1].
        best_scale_w_map (np.ndarray): optional width scale at max vote.
        best_scale_h_map (np.ndarray): optional height scale at max vote.

    Returns:
        peaks (list): [(px, py, angle_deg, scale_w, scale_h, score), ...] in crop coordinates.
                      scale_w/scale_h are 1.0 when the respective map is not provided.
    """
    rh, rw   = acc_norm.shape
    min_dist = int(min(card_w, card_h) * 0.4)
    acc_work = acc_norm.copy()
    peaks    = []

    for _ in range(n_cards):
        idx    = np.argmax(acc_work)
        py, px = np.unravel_index(idx, acc_work.shape)
        score  = float(acc_work[py, px])
        if score < threshold:
            break
        angle_deg = float(best_angle_map[py, px])
        scale_w   = float(best_scale_w_map[py, px]) if best_scale_w_map is not None else 1.0
        scale_h   = float(best_scale_h_map[py, px]) if best_scale_h_map is not None else 1.0
        peaks.append((int(px), int(py), angle_deg, scale_w, scale_h, score))

        y0, y1 = max(0, py - min_dist), min(rh, py + min_dist)
        x0, x1 = max(0, px - min_dist), min(rw, px + min_dist)
        acc_work[y0:y1, x0:x1] = 0

    return peaks
def detect_cards_in_blob(blob, labeled, img_shape, card_w, card_h,
                          single_area, n_angles=36, n_pts=80, threshold=0.25,
                          scales=None):
    """
    Full GHT pipeline for one multi-card blob.

    Chains: get_blob_contour → ght_accumulate → find_peaks,
    then converts peak coordinates from crop space to full image space.

    Args:
        blob        : RegionProperties of the multi-card blob.
        labeled     : integer label image.
        img_shape   : (H, W) of the full image.
        card_w/h    : nominal card dimensions in pixels.
        single_area : reference area of one card (card_w * card_h).
        n_angles    : number of angles over full 180° range.
        n_pts       : perimeter sample points per angle.
        threshold   : min normalized score to accept a detection.
        scales      : list of scale multipliers, e.g. [0.8, 0.9, 1.0, 1.1, 1.2].
                      Defaults to [1.0] (fixed size).

    Returns:
        cards       (list)      : [(cx, cy, angle_deg, scale_w, scale_h, score), ...] in full image coords.
        acc_norm    (np.ndarray): normalized accumulator for visualization.
        contour_img (np.ndarray): outer contour image used as input.
        offset      (tuple)     : (off_x, off_y) for coordinate conversion.
    """
    n_cards = max(2, round(blob.area / single_area))

    contour_img, (off_x, off_y), blob_mask = get_blob_contour(blob, labeled, img_shape)

    angles_deg = np.linspace(-90, 90, n_angles, endpoint=False).tolist()

    acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map = ght_accumulate(
        contour_img, card_w, card_h, angles_deg, n_pts,
        blob_mask=blob_mask, scales=scales)

    peaks = find_peaks(acc_norm, best_angle_map, n_cards, card_w, card_h,
                       threshold, best_scale_w_map=best_scale_w_map,
                       best_scale_h_map=best_scale_h_map)

    # Convert from crop coordinates to full image coordinates
    cards = [(px + off_x, py + off_y, ang, sw, sh, sc)
             for px, py, ang, sw, sh, sc in peaks]

    return cards, acc_norm, contour_img, (off_x, off_y)
 


