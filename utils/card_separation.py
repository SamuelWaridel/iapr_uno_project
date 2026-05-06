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

def classify_blobs(filled, single_area, overlap_ratio=1.3):
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
    solo_blobs  = [b for b in blobs if b.area <= overlap_ratio * single_area]
    multi_blobs = [b for b in blobs if b.area >  overlap_ratio * single_area]
    return labeled, solo_blobs, multi_blobs

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
    return contour_img, (c0, r0)

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
        r_vecs (np.ndarray): shape (n_pts, 2), dtype float32.
                             Each row (dx, dy) is the vector from
                             a perimeter sample to the card center.
    """
    a  = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    rot    = np.array([[ca, -sa], [sa, ca]])
    hw, hh = card_w / 2, card_h / 2
 
    # 4 corners of the card in local frame (center = origin)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    corners = (rot @ corners.T).T   # rotate to global frame
 
    pts_per_side = n_pts // 4
    perim_pts = []
    for i in range(4):
        p0, p1 = corners[i], corners[(i+1) % 4]
        ts  = np.linspace(0, 1, pts_per_side, endpoint=False)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        perim_pts.append(pts)
 
    perim_pts = np.vstack(perim_pts)      # (n_pts, 2)
    r_vecs    = -perim_pts.astype(np.float32)   # vector → center
    return r_vecs

def ght_accumulate(contour_img, card_w, card_h, angles_deg, n_pts=80):
    """
    Accumulate votes in 2D Hough space using the R-table.
 
    For each test angle:
      1. Get all contour pixel positions (xs, ys).
      2. Compute all candidate centers via broadcasting:
         centers = pixel_coords + r_vecs  →  shape (N_pixels, N_pts, 2)
      3. Cast votes with np.add.at (atomic, no race conditions).
      4. Keep the max across angles in the global accumulator.
 
    The full 180° range (-90° to +90°) covers:
      - Portrait cards   (angle ≈ 0°)
      - Landscape cards  (angle ≈ ±90°)
      - All tilted cards (any angle in between)
 
    Args:
        contour_img (np.ndarray): uint8 outer contour, shape (H, W).
        card_w/h    (int)       : card dimensions.
        angles_deg  (list)      : angles to test in degrees.
        n_pts       (int)       : perimeter sample points per angle.
 
    Returns:
        acc_norm       (np.ndarray): normalized accumulator [0-1], shape (H, W).
        best_angle_map (np.ndarray): float32, best angle in degrees per pixel.
    """
    rh, rw = contour_img.shape
    acc            = np.zeros((rh, rw), dtype=np.float32)
    best_angle_map = np.zeros((rh, rw), dtype=np.float32)
 
    ys, xs = np.where(contour_img > 0)
    if len(xs) == 0:
        return acc, best_angle_map
 
    # Contour pixel positions: shape (N, 2)
    pixel_coords = np.stack([xs, ys], axis=1).astype(np.float32)
 
    for angle_deg in angles_deg:
        r_vecs = build_r_vectors(card_w, card_h, angle_deg, n_pts)  # (M, 2)
 
        # All candidate centers at once: (N,1,2) + (1,M,2) → (N,M,2)
        centers = pixel_coords[:, None, :] + r_vecs[None, :, :]
 
        cx_all = centers[:, :, 0].ravel().astype(np.int32)
        cy_all = centers[:, :, 1].ravel().astype(np.int32)
 
        valid  = (cx_all >= 0) & (cx_all < rw) & (cy_all >= 0) & (cy_all < rh)
        cx_v, cy_v = cx_all[valid], cy_all[valid]
 
        acc_angle = np.zeros((rh, rw), dtype=np.float32)
        np.add.at(acc_angle, (cy_v, cx_v), 1.0)
 
        improved = acc_angle > acc
        acc[improved]            = acc_angle[improved]
        best_angle_map[improved] = angle_deg
 
    # Smooth to merge nearby votes
    acc = cv2.GaussianBlur(acc, (21, 21), 0)
    acc_norm = acc / (acc.max() + 1e-8)
    return acc_norm, best_angle_map


def find_peaks(acc_norm, best_angle_map, n_cards, card_w, card_h, threshold=0.25):
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
        acc_norm       (np.ndarray): normalized accumulator from ght_accumulate().
        best_angle_map (np.ndarray): angle at max vote, from ght_accumulate().
        n_cards        (int)       : expected number of cards in this blob.
        card_w/h       (int)       : card dimensions (used for suppression radius).
        threshold      (float)     : min score to accept a detection [0-1].
 
    Returns:
        peaks (list): [(px, py, angle_deg, score), ...] in crop coordinates.
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
        peaks.append((int(px), int(py), angle_deg, score))
 
        # Suppress neighborhood
        y0, y1 = max(0, py-min_dist), min(rh, py+min_dist)
        x0, x1 = max(0, px-min_dist), min(rw, px+min_dist)
        acc_work[y0:y1, x0:x1] = 0
 
    return peaks
def detect_cards_in_blob(blob, labeled, img_shape, card_w, card_h,
                          single_area, n_angles=36, n_pts=80, threshold=0.25):
    """
    Full GHT pipeline for one multi-card blob.
 
    Chains: get_blob_contour → ght_accumulate → find_peaks,
    then converts peak coordinates from crop space to full image space.
 
    Args:
        blob        : RegionProperties of the multi-card blob.
        labeled     : integer label image.
        img_shape   : (H, W) of the full image.
        card_w/h    : true card dimensions in pixels.
        single_area : reference area of one card (card_w * card_h).
        n_angles    : number of angles over full 180° range.
                      More = more precise but slower.
        n_pts       : perimeter sample points per angle.
        threshold   : min normalized score to accept a detection.
 
    Returns:
        cards       (list)      : [(cx, cy, angle_deg, score), ...] in full image coords.
        acc_norm    (np.ndarray): normalized accumulator for visualization.
        contour_img (np.ndarray): outer contour image used as input.
        offset      (tuple)     : (off_x, off_y) for coordinate conversion.
    """
    n_cards = max(2, round(blob.area / single_area))
 
    contour_img, (off_x, off_y) = get_blob_contour(blob, labeled, img_shape)
 
    angles_deg = np.linspace(-90, 90, n_angles, endpoint=False).tolist()
 
    acc_norm, best_angle_map = ght_accumulate(
        contour_img, card_w, card_h, angles_deg, n_pts)
 
    peaks = find_peaks(acc_norm, best_angle_map, n_cards, card_w, card_h, threshold)
 
    # Convert from crop coordinates to full image coordinates
    cards = [(px + off_x, py + off_y, ang, sc) for px, py, ang, sc in peaks]
 
    return cards, acc_norm, contour_img, (off_x, off_y)
 


