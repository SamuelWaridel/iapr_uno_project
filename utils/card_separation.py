import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter1d
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



def find_peaks(acc_norm, best_angle_map, card_w, card_h,
               threshold=0.2, best_scale_w_map=None, best_scale_h_map=None,
               exclude_peaks=None, cross_pass_radius=None):
    """
    Extract card center positions from the accumulator via greedy
    non-maximum suppression.

    Algorithm:
      1. Pre-suppress positions near any already-accepted peak (cross-pass NMS).
      2. Find global max → card center candidate.
      3. If score < threshold, stop.
      4. Suppress a neighborhood of radius min(card_w,card_h)*0.4
         around the peak (within-pass NMS).
      5. Repeat until no peak remains above threshold.

    Args:
        acc_norm          (np.ndarray): normalized accumulator.
        best_angle_map    (np.ndarray): float32, best angle per pixel (°).
        card_w/h          (int)       : nominal card dimensions.
        threshold         (float)     : min score to accept a detection.
        best_scale_w_map  (np.ndarray): optional width scale at max vote.
        best_scale_h_map  (np.ndarray): optional height scale at max vote.
        exclude_peaks     (list)      : peaks from previous passes —
                                        [(px, py, ...), ...].  Regions around
                                        these are zeroed before NMS starts.
        cross_pass_radius (int)       : suppression radius for exclude_peaks.
                                        Defaults to min(card_w,card_h)*0.25.

    Returns:
        peaks (list): [(px, py, angle_deg, scale_w, scale_h, score), ...]
    """
    rh, rw        = acc_norm.shape
    within_radius = int(min(card_w, card_h) * 0.8)
    if cross_pass_radius is None:
        cross_pass_radius = int(min(card_w, card_h) * 0.45)
    acc_work = acc_norm.copy()

    # Cross-pass suppression: zero out neighbourhoods of previous-pass detections
    if exclude_peaks:
        for ep in exclude_peaks:
            epx, epy = int(ep[0]), int(ep[1])
            y0, y1 = max(0, epy - cross_pass_radius), min(rh, epy + cross_pass_radius)
            x0, x1 = max(0, epx - cross_pass_radius), min(rw, epx + cross_pass_radius)
            acc_work[y0:y1, x0:x1] = 0

    peaks = []
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

        y0, y1 = max(0, py - within_radius), min(rh, py + within_radius)
        x0, x1 = max(0, px - within_radius), min(rw, px + within_radius)
        acc_work[y0:y1, x0:x1] = 0

    return peaks


def _refine_angle(feature_img, template, px, py, coarse_angle_deg,
                  search_range=6, fine_step=1.0):
    """
    Refine the card rotation angle around a coarse estimate.

    Evaluates TM_CCORR_NORMED at position (px, py) for angles in
    [coarse_angle_deg ± search_range] with fine_step resolution.

    Args:
        feature_img      : (H, W) uint8 white-edge image.
        template         : (CARD_H, CARD_W) uint8 binary edge template.
        px, py           : detected card center in feature_img coordinates.
        coarse_angle_deg : angle from the coarse search (degrees).
        search_range     : ± degrees to search around coarse_angle_deg.
        fine_step        : angular resolution of the fine search (degrees).

    Returns:
        best_angle (float): refined angle in degrees.
    """
    fh, fw = feature_img.shape
    feat_f = feature_img.astype(np.float32)

    angles = np.arange(coarse_angle_deg - search_range,
                       coarse_angle_deg + search_range + fine_step,
                       fine_step)

    best_score = -1.0
    best_angle = coarse_angle_deg

    for ang in angles:
        rot_tmpl, (tw, th) = _rotate_template(template, ang)
        if tw >= fw or th >= fh:
            continue

        x0, y0 = px - tw // 2, py - th // 2
        x1, y1 = x0 + tw,      y0 + th
        if x0 < 0 or y0 < 0 or x1 > fw or y1 > fh:
            continue

        patch  = feat_f[y0:y1, x0:x1]
        tmpl_f = rot_tmpl.astype(np.float32)
        num    = float(np.dot(patch.ravel(), tmpl_f.ravel()))
        denom  = float(np.sqrt(np.sum(patch ** 2) * np.sum(tmpl_f ** 2))) + 1e-8
        score  = num / denom

        if score > best_score:
            best_score = score
            best_angle = ang

    return float(best_angle)


def _mask_detected_card(feature_img, cx_crop, cy_crop, angle_deg, sw, sh,
                         card_w, card_h, margin=20):
    """Zero out the oriented rectangle of a detected card in feature_img (in-place)."""
    hw = card_w * sw / 2 + margin
    hh = card_h * sh / 2 + margin
    corners_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
                              dtype=np.float32)
    a  = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    corners = ((np.array([[ca, -sa], [sa, ca]]) @ corners_local.T).T
               + np.array([cx_crop, cy_crop])).astype(np.int32)
    cv2.fillPoly(feature_img, [corners], 0)




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


def template_match_accumulate(feature_img, template, angles_deg, scales=None,
                               norm_max=None):
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
        norm_max    (float)     : if provided, normalise by this value instead of
                                  the current accumulator's max.  Pass the raw_max
                                  from the first pass so that multi-pass thresholds
                                  are on the same absolute scale.

    Returns:
        acc_norm         (np.ndarray): normalised accumulator [0-1], (H, W).
        best_angle_map   (np.ndarray): float32, best angle per pixel (°).
        best_scale_w_map (np.ndarray): float32, best width  scale per pixel.
        best_scale_h_map (np.ndarray): float32, best height scale per pixel.
        raw_max          (float)     : this pass's raw accumulator maximum
                                       (use as norm_max for subsequent passes).
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

    # Gaussian blur to collapse flat plateaus into a single sharp peak
    sigma = int(min(CARD_W, CARD_H) * 0.05)
    if sigma > 0:
        acc = cv2.GaussianBlur(acc, (0, 0), sigma)

    raw_max  = acc.max() + 1e-8
    divisor  = norm_max if norm_max is not None else raw_max
    acc_norm = acc / divisor
    return acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map, raw_max


def detect_cards_in_blob(blob, labeled, img_shape, card_w, card_h,
                          n_angles=72, threshold=0.25,
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
        
    thresholds    = threshold if isinstance(threshold, (list, tuple)) else [threshold]
    feature_work  = feature_img.copy()
    all_peaks     = []
    acc_norm      = None
    first_raw_max = None

    for thr in thresholds:
        if template is not None:
            acc_norm, best_angle_map, best_scale_w_map, best_scale_h_map, raw_max = \
                template_match_accumulate(feature_work, template, angles_deg, scales,
                                         norm_max=first_raw_max)
            if first_raw_max is None:
                first_raw_max = raw_max
        else:
            break

        new_peaks = find_peaks(acc_norm, best_angle_map, card_w, card_h,
                               thr, best_scale_w_map=best_scale_w_map,
                               best_scale_h_map=best_scale_h_map,
                               exclude_peaks=all_peaks)

        for i, (px, py, ang, sw_val, sh_val, sc) in enumerate(new_peaks):
            if template is not None:
                refined_ang = _refine_angle(feature_work, template, px, py, ang)
                new_peaks[i] = (px, py, refined_ang, sw_val, sh_val, sc)
            _mask_detected_card(feature_work, px, py, new_peaks[i][2], sw_val, sh_val,
                                card_w, card_h)
        all_peaks.extend(new_peaks)

        if not new_peaks:
            break

    cards = [(px + off_x, py + off_y, ang, sw, sh, sc)
             for px, py, ang, sw, sh, sc in all_peaks]

    return cards, acc_norm, feature_img, (off_x, off_y)
 


