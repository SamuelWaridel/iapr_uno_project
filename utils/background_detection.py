import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv


def detect_background_type(image, saturation_threshold=0.15, corner_size_ratio=0.10):
    """
    Detect whether the input image has a uniform white/grey background or a
    colored, patterned ("noisy") background.

    Strategy:
        Cards are always placed inside the play area, never in the corners of
        the photograph. We therefore sample the four corner patches and look
        at their median HSV saturation:
          - white/grey table  -> very low saturation (typically < 0.05)
          - colored leaf bg   -> high saturation (typically > 0.30)
        A single threshold around 0.15 separates the two regimes cleanly.

    Args:
        image (np.ndarray): RGB image, shape (H, W, 3).
                            Values either in [0, 255] (uint8) or [0, 1] (float).
        saturation_threshold (float): median-saturation threshold separating
                            'white' from 'noisy'. Default 0.15.
        corner_size_ratio (float): fraction of the image height/width used
                            for each corner patch. Default 0.10 (=10%).

    Returns:
        bg_type    (str)  : 'white' if uniform light background,
                            'noisy' if colored / patterned background.
        median_sat (float): median saturation in the four corners.
                            Useful for debugging or threshold tuning.
    """
    # Normalize to [0, 1] if needed (rgb2hsv expects floats in [0, 1])
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    H, W = img.shape[:2]
    ch = max(1, int(H * corner_size_ratio))
    cw = max(1, int(W * corner_size_ratio))

    # Stack the four corners into a single (4*ch, cw, 3) image so rgb2hsv
    # only has to be called once.
    corners = np.vstack([
        np.hstack([img[:ch,  :cw],  img[:ch,  -cw:]]),
        np.hstack([img[-ch:, :cw],  img[-ch:, -cw:]]),
    ])

    sat = rgb2hsv(corners)[:, :, 1]
    median_sat = float(np.median(sat))

    bg_type = "white" if median_sat < saturation_threshold else "noisy"
    return bg_type, median_sat


def visualize_background_detection(image, saturation_threshold=0.15,
                                   corner_size_ratio=0.10):
    """
    Show the four corner patches used by `detect_background_type` together with
    the predicted background type. Useful for sanity-checking the threshold.

    Args:
        image (np.ndarray): RGB image, shape (H, W, 3).
        saturation_threshold (float): same as in `detect_background_type`.
        corner_size_ratio (float): same as in `detect_background_type`.
    """
    bg_type, median_sat = detect_background_type(
        image, saturation_threshold, corner_size_ratio
    )

    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    H, W = img.shape[:2]
    ch = max(1, int(H * corner_size_ratio))
    cw = max(1, int(W * corner_size_ratio))

    corners = {
        "top-left":     img[:ch,  :cw],
        "top-right":    img[:ch,  -cw:],
        "bottom-left":  img[-ch:, :cw],
        "bottom-right": img[-ch:, -cw:],
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"prediction: {bg_type}\nmedian corner sat = {median_sat:.3f}")
    axes[0].axis("off")
    for ax, (name, patch) in zip(axes[1:], corners.items()):
        ax.imshow(patch)
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def validate_background_detector(images_dict, saturation_threshold=0.15,
                                 corner_size_ratio=0.10, verbose=False):
    """
    Validate `detect_background_type` against the ground truth read from the
    filenames (naming convention: the 3rd character from the end is '7' or '8'
    for white background, '9' for noisy background).

    Args:
        images_dict (dict): {filename: np.ndarray} as returned by
                            load_image_subset / load_all_images.
        saturation_threshold (float): forwarded to detect_background_type.
        corner_size_ratio  (float)  : forwarded to detect_background_type.
        verbose (bool): if True, print every misclassified image.

    Returns:
        accuracy       (float): overall accuracy in [0, 1].
        misclassified  (list) : list of (filename, true_label, pred_label, median_sat).
    """
    correct = 0
    total = 0
    misclassified = []

    for filename, img in images_dict.items():
        # Strip the .jpg extension to get the image id, then read the convention digit.
        image_id = filename[:-4] if filename.endswith(".jpg") else filename
        digit = image_id[-3]
        if digit in ("7", "8"):
            true_label = "white"
        elif digit == "9":
            true_label = "noisy"
        else:
            # Unknown convention: skip
            continue

        pred_label, median_sat = detect_background_type(
            img, saturation_threshold, corner_size_ratio
        )
        total += 1
        if pred_label == true_label:
            correct += 1
        else:
            misclassified.append((filename, true_label, pred_label, median_sat))
            if verbose:
                print(f"  MISS  {filename:>15s}  true={true_label:5s} "
                      f"pred={pred_label:5s}  median_sat={median_sat:.3f}")

    accuracy = correct / total if total > 0 else float("nan")
    print(f"Background detection accuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy, misclassified
