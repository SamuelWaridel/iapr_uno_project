import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from PIL import Image
import cv2

CARD_W      = 353
CARD_H      = 576
SINGLE_AREA = CARD_W * CARD_H

# White oval inscribed inside the card (fraction of card dimensions).
# These match the central white ellipse visible on every UNO card.
OVAL_W_RATIO = 0.58   # oval width  / card width
OVAL_H_RATIO = 0.60   # oval height / card height

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

def extract_white_features(blob, img_rgb, labeled, img_shape,
                            white_sat_max=0.25, white_val_min=0.70,
                            smooth_sigma=2, margin_ratio=0.4):
    """
    Extract edges of white regions (card frame + central oval) inside a blob.

    UNO cards have two prominent white structures visible in any colour image:
      1. A thin white rectangular frame around the card edge.
      2. A large white oval at the card centre.

    Both appear as bright (low-S, high-V) regions in HSV space.  By taking
    the edges of their union we get a rich set of feature points that covers
    both the outer card boundary *and* the internal oval — providing far more
    GHT votes than the blob outer contour alone, and surviving partial
    occlusion because the oval of an occluded card is often still fully visible.

    Args:
        blob          : RegionProperties of the blob.
        img_rgb       : (H, W, 3) uint8 original image.
        labeled       : integer label image from classify_blobs().
        img_shape     : (H, W) of the full image.
        white_sat_max : HSV saturation upper bound (white is unsaturated).
        white_val_min : HSV value lower bound (white is bright).
        smooth_sigma  : 2D Gaussian sigma applied to the white mask before
                        edge extraction to remove salt-and-pepper noise.
        margin_ratio  : padding around the blob bounding box.

    Returns:
        edge_img    (np.ndarray): uint8 binary edge image (255=edge, 0=bg).
                                  Used as contour_img in ght_accumulate().
        white_mask  (np.ndarray): uint8 binary white mask (255=white, 0=bg).
                                  Used as blob_mask for Sobel gradient computation
                                  (replaces blob_mask_filled from get_blob_contour).
        offset      (tuple)     : (off_x, off_y) top-left in full image coords.
    """
    from scipy.ndimage import gaussian_filter

    margin = int(max(CARD_W, CARD_H) * margin_ratio)
    minr, minc, maxr, maxc = blob.bbox
    r0 = max(0, minr - margin); r1 = min(img_shape[0], maxr + margin)
    c0 = max(0, minc - margin); c1 = min(img_shape[1], maxc + margin)

    blob_region = (labeled[r0:r1, c0:c1] == blob.label)
    crop_hsv    = rgb2hsv(img_rgb[r0:r1, c0:c1].astype(np.float32) / 255.0)

    raw_white = ((crop_hsv[:, :, 1] < white_sat_max) &
                 (crop_hsv[:, :, 2] > white_val_min) &
                 blob_region)

    if smooth_sigma > 0:
        smoothed   = gaussian_filter(raw_white.astype(np.float32), smooth_sigma)
        white_mask = (smoothed > 0.5).astype(np.uint8) * 255
    else:
        white_mask = raw_white.astype(np.uint8) * 255

    # RETR_LIST captures both outer (frame) and inner (oval boundary) edges
    contours, _ = cv2.findContours(white_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edge_img = np.zeros_like(white_mask)
    cv2.drawContours(edge_img, contours, -1, 255, thickness=2)

    return edge_img, white_mask, (c0, r0)


def get_blob_contour(blob, labeled, img_shape, margin_ratio=0.4, smooth_sigma=3):
    """
    Extract only the outer contour of a blob as a binary image.

    Using RETR_EXTERNAL ensures we capture only the blob boundary
    and not any internal structure (card numbers, oval pattern, etc.).
    This clean signal is what the GHT will vote against.

    A 1D Gaussian is applied to the (x, y) coordinates of each contour
    point (with wrap-around since the contour is closed). This removes
    high-frequency noise along straight sides while leaving corners intact:
    corners are low-frequency large-scale features that survive small-sigma
    smoothing, whereas side noise is high-frequency and gets suppressed.

    Args:
        blob         : RegionProperties of the blob.
        labeled      : integer label image from classify_blobs().
        img_shape    : (H, W) of the full image.
        margin_ratio : fraction of max(card_w, card_h) to add as padding.
        smooth_sigma : Gaussian sigma (in contour-point units) for 1D
                       coordinate smoothing. 3 removes ~3 px of side noise.
                       Set to 0 to disable.

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

    if smooth_sigma > 0:
        smoothed = []
        for c in contours:
            pts = c[:, 0, :].astype(float)          # (N, 2)
            pts[:, 0] = gaussian_filter1d(pts[:, 0], smooth_sigma, mode='wrap')
            pts[:, 1] = gaussian_filter1d(pts[:, 1], smooth_sigma, mode='wrap')
            smoothed.append(pts.round().astype(np.int32)[:, None, :])
        contours = smoothed

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


def build_r_vectors_combined(card_w, card_h, angle_deg,
                              oval_w_ratio=OVAL_W_RATIO, oval_h_ratio=OVAL_H_RATIO,
                              n_pts=120):
    """
    R-table combining the outer card rectangle and the inner white oval.

    The white oval is the central ellipse printed on every UNO card.
    By including its perimeter in the R-table, every white pixel on the
    oval boundary also votes for the card centre — giving a strong, stable
    signal that survives partial occlusion by an overlapping card.

    Geometry (local frame, centre = origin, before card rotation):
      Rectangle : ±hw × ±hh  (hw = card_w/2, hh = card_h/2)
      Oval      : parametric (a·cos t, b·sin t)
                  a = card_w * oval_w_ratio / 2
                  b = card_h * oval_h_ratio / 2

    Inward normals:
      Rectangle sides : same 4 directions as build_r_vectors.
      Oval perimeter  : (-cos t, -sin t) rotated by card angle
                        (circular approximation — valid since a ≈ b,
                         sufficient precision for 8-bin angle filtering).

    Args:
        card_w/h      (int)  : nominal card dimensions.
        angle_deg     (float): card rotation angle in degrees.
        oval_w_ratio  (float): oval full-width  / card_w.
        oval_h_ratio  (float): oval full-height / card_h.
        n_pts         (int)  : total sample points (split evenly rect/oval).

    Returns:
        r_vecs      (np.ndarray): shape (n_pts, 2), float32.
        grad_angles (np.ndarray): shape (n_pts,),   float32, inward normal angles.
    """
    n_rect = n_pts // 2
    n_oval = n_pts - n_rect

    # --- Outer rectangle ---
    r_vecs_rect, grad_rect = build_r_vectors(card_w, card_h, angle_deg, n_rect)

    # --- Inner oval ---
    a_rad = np.deg2rad(angle_deg)
    ca, sa = np.cos(a_rad), np.sin(a_rad)
    rot = np.array([[ca, -sa], [sa, ca]])

    oa = card_w * oval_w_ratio / 2   # semi-axis along card width
    ob = card_h * oval_h_ratio / 2   # semi-axis along card height

    ts = np.linspace(0, 2 * np.pi, n_oval, endpoint=False)

    # Oval perimeter in local frame: (oa·cos t, ob·sin t)
    oval_local = np.column_stack([oa * np.cos(ts), ob * np.sin(ts)])  # (N, 2)
    oval_rot   = (rot @ oval_local.T).T                               # (N, 2) rotated

    # R-vectors: from oval edge to card centre = -oval_rot
    r_vecs_oval = -oval_rot.astype(np.float32)

    # Inward normal in local frame: (-cos t, -sin t)  [circular approx]
    n_local = np.column_stack([-np.cos(ts), -np.sin(ts)])
    n_rot   = (rot @ n_local.T).T
    grad_oval = np.arctan2(n_rot[:, 1], n_rot[:, 0]).astype(np.float32)

    r_vecs      = np.vstack([r_vecs_rect, r_vecs_oval])
    grad_angles = np.concatenate([grad_rect, grad_oval])
    return r_vecs, grad_angles


def build_r_table_from_template(template, angle_deg, sw=1.0, sh=1.0, n_pts=200):
    """
    Build a GHT R-table from the binary edge template produced by
    build_card_template() in utils/card_template.py.

    The template is a (CARD_H, CARD_W) uint8 image whose white pixels
    trace the inner frame edge and the outer oval edge of a UNO card at
    standard portrait orientation (0°).  For each detection angle we:
      1. Subsample n_pts edge pixels evenly from the template.
      2. Compute the r_vec = (card_centre − pixel) for each sample,
         then apply the anisotropic scale (sw, sh).
      3. Estimate gradient angles via Sobel on a dilated copy of the
         template (dilation stabilises the gradient on thin lines).
      4. Rotate r_vecs and grad_angles by angle_deg.

    Args:
        template  (np.ndarray): (CARD_H, CARD_W) uint8 binary edge image.
        angle_deg (float)     : card rotation angle in degrees.
        sw, sh    (float)     : width / height scale factors (default 1.0).
        n_pts     (int)       : number of sample points to keep.

    Returns:
        r_vecs      (np.ndarray): (N, 2) float32 — vector from sample to card centre.
        grad_angles (np.ndarray): (N,)   float32 — gradient angle (radians) at each sample.
    """
    ys_all, xs_all = np.where(template > 0)
    if len(xs_all) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Evenly subsample to at most n_pts points
    if len(xs_all) > n_pts:
        idx = np.round(np.linspace(0, len(xs_all) - 1, n_pts)).astype(int)
        xs, ys = xs_all[idx], ys_all[idx]
    else:
        xs, ys = xs_all, ys_all

    # Card centre in the template image (portrait, 0°)
    cx0, cy0 = CARD_W / 2.0, CARD_H / 2.0

    # R-vectors at 0°: (centre − sample), scaled anisotropically
    dx = (cx0 - xs) * sw
    dy = (cy0 - ys) * sh

    # Gradient angles: Sobel on a slightly dilated template for stability
    dilated = cv2.dilate(template, np.ones((3, 3), np.uint8), iterations=1)
    gx_img  = cv2.Sobel(dilated.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    gy_img  = cv2.Sobel(dilated.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    grad_angles = np.arctan2(gy_img[ys, xs], gx_img[ys, xs]).astype(np.float32)

    # Rotate r_vecs and grad_angles by angle_deg
    if angle_deg != 0.0:
        a_rad   = np.deg2rad(angle_deg)
        ca, sa  = np.cos(a_rad), np.sin(a_rad)
        rot     = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        rv      = (rot @ np.stack([dx, dy])).T   # (N, 2)
        dx, dy  = rv[:, 0], rv[:, 1]
        grad_angles = np.arctan2(
            np.sin(grad_angles + a_rad),
            np.cos(grad_angles + a_rad),
        ).astype(np.float32)

    r_vecs = np.stack([dx, dy], axis=1).astype(np.float32)
    return r_vecs, grad_angles


def ght_accumulate(contour_img, card_w, card_h, angles_deg, n_pts=80,
                   blob_mask=None, n_angle_bins=8, scales=None,
                   use_oval=False, template=None):
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
    independent width/height multipliers.

    Args:
        contour_img   (np.ndarray): uint8 feature edge image, shape (H, W).
                                    Can be the blob outer contour (legacy) or
                                    the white-feature edge image from
                                    extract_white_features() (preferred).
        card_w/h      (int)       : nominal card dimensions.
        angles_deg    (list)      : angles to test in degrees.
        n_pts         (int)       : perimeter sample points per angle.
        blob_mask     (np.ndarray): uint8 filled mask used for Sobel gradients.
                                    Pass the white_mask from extract_white_features()
                                    when use_oval=True.
        n_angle_bins  (int)       : gradient direction bins (default 8 = 45°/bin).
        scales        (list)      : scale multipliers. Each entry:
                                    - float s   → isotropic (s, s)
                                    - (sw, sh)  → anisotropic width/height scales
                                    Defaults to [(1.0, 1.0)] (fixed size).
        use_oval      (bool)      : if True, use build_r_vectors_combined()
                                    (rectangle + oval R-table) instead of the
                                    rectangle-only build_r_vectors().
                                    Ignored when `template` is provided.
        template      (np.ndarray): optional (CARD_H, CARD_W) uint8 binary edge
                                    image from build_card_template().  When
                                    provided, the R-table is built from the
                                    template (data-driven) instead of the
                                    analytical rectangle / oval formulas.
                                    Recommended: gives more accurate r_vecs.

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

    # Gradient angles at contour pixels — computed from the filled mask
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

    r_table_fn = build_r_vectors_combined if use_oval else build_r_vectors

    for sw, sh in norm_scales:
        w = int(card_w * sw)
        h = int(card_h * sh)

        for angle_deg in angles_deg:
            if template is not None:
                r_vecs, r_grad_angles = build_r_table_from_template(
                    template, angle_deg, sw=sw, sh=sh, n_pts=n_pts)
            else:
                r_vecs, r_grad_angles = r_table_fn(w, h, angle_deg, n_pts)
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


def find_peaks(acc_norm, best_angle_map, card_w, card_h,
               threshold=0.2, best_scale_w_map=None, best_scale_h_map=None):
    """
    Extract card center positions from the accumulator via greedy
    non-maximum suppression.

    Algorithm:
      1. Find global max → card center candidate.
      2. If score < threshold, stop.
      3. Suppress a neighborhood of radius min(card_w,card_h)*0.4
         around the peak (NMS: avoids picking the same card at different angles).
      4. Repeat until no peak remains above threshold.

    Args:
        acc_norm         (np.ndarray): normalized accumulator from ght_accumulate().
        best_angle_map   (np.ndarray): angle at max vote, from ght_accumulate().
        card_w/h         (int)       : nominal card dimensions (suppression radius).
        threshold        (float)     : min absolute score to accept a detection.
        best_scale_w_map (np.ndarray): optional width scale at max vote.
        best_scale_h_map (np.ndarray): optional height scale at max vote.

    Returns:
        peaks (list): [(px, py, angle_deg, scale_w, scale_h, score), ...]
    """
    rh, rw   = acc_norm.shape
    min_dist = int(min(card_w, card_h) * 0.4)
    acc_work = acc_norm.copy()
    peaks    = []

    while True:
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
def detect_ovals_in_blob(blob, img_rgb, labeled, img_shape,
                          card_w=CARD_W, card_h=CARD_H,
                          white_sat_max=0.25, white_val_min=0.70,
                          min_area_ratio=0.08, max_area_ratio=0.45,
                          min_ecc=0.40, max_ecc=0.95):
    """
    Detect the white ovals inside a multi-card blob.

    Each UNO card contains a large white ellipse at its centre. In HSV space
    white regions have low saturation and high value, making them easy to
    isolate within the blob area. Each detected oval gives:
      - centroid  → card centre (more reliable than GHT on blob contours)
      - orientation → card rotation angle

    The white card frame is much smaller in area and filtered out by the
    area thresholds. Number glyphs and text are too small or too circular.

    Args:
        blob           : RegionProperties of the multi-card blob.
        img_rgb        : (H, W, 3) uint8 original image.
        labeled        : integer label image from classify_blobs().
        img_shape      : (H, W) of the full image.
        card_w/h       : nominal card dimensions in pixels.
        white_sat_max  : HSV saturation upper bound for "white".
        white_val_min  : HSV value lower bound for "white".
        min_area_ratio : minimum oval area as a fraction of one card area.
        max_area_ratio : maximum oval area as a fraction of one card area.
        min_ecc/max_ecc: eccentricity bounds (0 = circle, 1 = line).

    Returns:
        cards  (list) : [(cx, cy, angle_deg, 1.0, 1.0, score), ...] in full image coords.
                        score is the oval area normalised to [0, 1].
        offset (tuple): (off_x, off_y) top-left of the crop in full image coords.
    """
    margin = int(max(card_w, card_h) * 0.4)
    minr, minc, maxr, maxc = blob.bbox
    r0 = max(0, minr - margin); r1 = min(img_shape[0], maxr + margin)
    c0 = max(0, minc - margin); c1 = min(img_shape[1], maxc + margin)

    blob_region = (labeled[r0:r1, c0:c1] == blob.label)
    crop_hsv    = rgb2hsv(img_rgb[r0:r1, c0:c1].astype(np.float32) / 255.0)

    white_mask = ((crop_hsv[:, :, 1] < white_sat_max) &
                  (crop_hsv[:, :, 2] > white_val_min) &
                  blob_region)

    labeled_white = label(white_mask)
    regions       = regionprops(labeled_white)

    single_area = card_w * card_h
    min_area    = min_area_ratio * single_area
    max_area    = max_area_ratio * single_area

    candidates = []
    for r in regions:
        if not (min_area <= r.area <= max_area):
            continue
        if r.minor_axis_length < 1:
            continue
        if not (min_ecc <= r.eccentricity <= max_ecc):
            continue
        candidates.append(r)

    if not candidates:
        return [], (c0, r0)

    max_a = max(r.area for r in candidates)
    cards = []
    for r in candidates:
        cy_crop, cx_crop = r.centroid
        # regionprops.orientation: angle from row-axis (vertical) to major axis, CCW.
        # portrait card (tall oval) → orientation ≈ 0 → card_angle = 0°
        # landscape card (wide oval) → orientation ≈ ±pi/2 → card_angle = ±90°
        angle_deg = float(np.rad2deg(r.orientation))
        score     = r.area / max_a
        cards.append((int(cx_crop + c0), int(cy_crop + r0), angle_deg, 1.0, 1.0, score))

    return cards, (c0, r0)


def _rotate_template(template, angle_deg):
    """
    Rotate a binary template image by angle_deg degrees.

    The output canvas is expanded so the rotated template fits without
    clipping, and the background is filled with zeros (black).

    Args:
        template  (np.ndarray): (H, W) uint8 binary image.
        angle_deg (float)     : rotation angle in degrees.
                                Positive = CCW in image coords (y-down).

    Returns:
        rotated (np.ndarray): (new_H, new_W) uint8 rotated template.
        (new_w, new_h) (tuple): dimensions of the rotated image.
    """
    h, w = template.shape
    cx, cy = w / 2.0, h / 2.0

    cos_a  = abs(np.cos(np.deg2rad(angle_deg)))
    sin_a  = abs(np.sin(np.deg2rad(angle_deg)))
    new_w  = int(np.ceil(h * sin_a + w * cos_a))
    new_h  = int(np.ceil(h * cos_a + w * sin_a))

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rotated = cv2.warpAffine(template, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)
    return rotated, (new_w, new_h)


def template_match_accumulate(feature_img, template, angles_deg, scales=None):
    """
    Build a detection accumulator via direct template matching (cross-correlation).

    For each (scale, angle) pair, resize the template by (sw, sh). Rotate to angle_deg.
    Slide the rotated template over feature_img using normalised cross-correlation (cv2.TM_CCORR_NORMED).
    Place each correlation score at the CENTRE position of the template in the accumulator (top-left → centre shift).
    Keep the best (max) score and corresponding angle / scale maps.

    After all pairs, the accumulator is Gaussian-blurred and normalised.

    Args:
        feature_img (np.ndarray): (H, W) uint8 white-edge image (query).
        template    (np.ndarray): (CARD_H, CARD_W) uint8 binary edge template.
        angles_deg  (list)      : rotation angles to test, in degrees.
        scales      (list)      : scale multipliers — float s → (s, s),
                                  or (sw, sh) for anisotropic.

    Returns:
        acc_norm         (np.ndarray): normalised accumulator [0-1], (H, W).
        best_angle_map   (np.ndarray): float32, best angle per pixel (°).
        best_scale_w_map (np.ndarray): float32, best width  scale per pixel.
        best_scale_h_map (np.ndarray): float32, best height scale per pixel.
    """
    if scales is None:
        scales = [1.0]
    norm_scales = [(s, s) if isinstance(s, (int, float)) else tuple(s)
                   for s in scales]

    rh, rw = feature_img.shape
    acc              = np.zeros((rh, rw), dtype=np.float32)
    best_angle_map   = np.zeros((rh, rw), dtype=np.float32)
    best_scale_w_map = np.ones((rh, rw),  dtype=np.float32)
    best_scale_h_map = np.ones((rh, rw),  dtype=np.float32)

    feat_f = feature_img.astype(np.float32)

    for sw, sh in norm_scales:
        tw_s = int(CARD_W * sw)
        th_s = int(CARD_H * sh)
        scaled = cv2.resize(template, (tw_s, th_s),
                            interpolation=cv2.INTER_LINEAR)

        for angle_deg in angles_deg:
            rot_tmpl, (tw, th) = _rotate_template(scaled, angle_deg)

            # Template must be strictly smaller than the feature image
            if tw >= rw or th >= rh:
                continue

            corr = cv2.matchTemplate(feat_f, rot_tmpl.astype(np.float32),
                                     cv2.TM_CCORR_NORMED)   # (rh-th+1, rw-tw+1)

            ch, cw = corr.shape
            oy, ox = th // 2, tw // 2   # offset: top-left → centre

            region = acc[oy:oy + ch, ox:ox + cw]
            improved = corr > region
            region[improved] = corr[improved] # numpy view, write also in acc 
            best_angle_map  [oy:oy + ch, ox:ox + cw][improved] = angle_deg
            best_scale_w_map[oy:oy + ch, ox:ox + cw][improved] = sw
            best_scale_h_map[oy:oy + ch, ox:ox + cw][improved] = sh

    raw_max  = acc.max() + 1e-8          # preserve absolute correlation scale
    acc_norm = acc / raw_max             # threshold now meaningful as absolute corr score
    return acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map


def detect_cards_in_blob(blob, labeled, img_shape, card_w, card_h,
                          n_angles=36, threshold=0.25,
                          scales=None, img_rgb=None, template=None):
    """
    Detect card centres in a multi-card blob 

    Args:
        blob        : RegionProperties of the multi-card blob.
        labeled     : integer label image.
        img_shape   : (H, W) of the full image.
        card_w/h    : nominal card dimensions in pixels.
        single_area : reference area of one card (card_w * card_h).
        n_angles    : number of angles over full 180° range.
        n_pts       : total R-table sample points (split rect/oval when use_oval).
        threshold   : min normalized score to accept a detection.
        scales      : scale multipliers, e.g. [0.9, 1.0, 1.1].
        img_rgb     : (H, W, 3) uint8 original image.
                      When provided, uses extract_white_features() and the
                      combined rect+oval R-table. Strongly recommended.
        template    : optional (CARD_H, CARD_W) uint8 binary edge image from
                      build_card_template().  When provided, uses
                      template_match_accumulate() (direct cross-correlation)
                      instead of the GHT vote accumulator.

    Returns:
        cards       (list)      : [(cx, cy, angle_deg, scale_w, scale_h, score), ...]
                                  in full image coordinates.
        acc_norm    (np.ndarray): normalised accumulator for visualisation.
        feature_img (np.ndarray): white-edge image fed to the accumulator.
        offset      (tuple)     : (off_x, off_y) top-left of the crop.
    """
    angles_deg = np.linspace(-90, 90, n_angles, endpoint=False).tolist()

    # Extract white-feature edge image
    if img_rgb is not None:
        feature_img, white_mask, (off_x, off_y) = extract_white_features(
            blob, img_rgb, labeled, img_shape)
    else:
        feature_img, (off_x, off_y), white_mask = get_blob_contour(
            blob, labeled, img_shape)
        
    if template is not None:
        # Direct template matching: slide rotated template over feature_img
        acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map = \
            template_match_accumulate(feature_img, template, angles_deg, scales)

    peaks = find_peaks(acc_norm, best_angle_map, card_w, card_h,
                       threshold, best_scale_w_map=best_scale_w_map,
                       best_scale_h_map=best_scale_h_map)

    cards = [(px + off_x, py + off_y, ang, sw, sh, sc)
             for px, py, ang, sw, sh, sc in peaks]

    return cards, acc_norm, feature_img, (off_x, off_y)
 


