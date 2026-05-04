import numpy as np
from skimage.color import rgb2hsv
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.group_detect import *


def blob_bbox(blobs, centroids_xy, areas):
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
     
    # --- Union-Find helpers ---
    parent = list(range(len(blobs)))
 
    def find(i):
        # Path compression: flatten the tree on the way up
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
 
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
 
        # Enclosing bbox: smallest box that covers all blobs in the cluster
        minr = min(blobs[i].bbox[0] for i in idxs)
        minc = min(blobs[i].bbox[1] for i in idxs)
        maxr = max(blobs[i].bbox[2] for i in idxs)
        maxc = max(blobs[i].bbox[3] for i in idxs)
 
        group_centroids[idxs[0]] = (cx, cy)
        group_bboxes[idxs[0]]    = (minr, minc, maxr, maxc)
 
    return group_centroids, group_bboxes

def compute_color_features(card):
    """
    Compute the mean of each channel in the colored portion of the card.
    Determines the region of interest based on the quantile of the ratio in each channel.
    Works best with single-card images where the card occupies a significant portion of the image.

    Args:
        card (numpy.ndarray): The input card image. Must be a 3-channel RGB image.

    Returns:
        means (dict): A dictionary containing the mean of the color ratios.
    """
    # select colored region based on saturation
    sat_mask = rgb2hsv(card / 255.0)[:, :, 1] > 0.3
    
    r_g = card[:,:,0][sat_mask] / (card[:,:,1][sat_mask] + 1e-8)  # avoid division by zero
    r_b = card[:,:,0][sat_mask] / (card[:,:,2][sat_mask] + 1e-8)
    g_b = card[:,:,1][sat_mask] / (card[:,:,2][sat_mask] + 1e-8)
    
    means = {"r/g": r_g.mean(), "r/b": r_b.mean(), "g/b": g_b.mean()}
            
    return means

def determine_card_color(color_means, tolerance=0.25):
    """
    Determine card color based on the ratios of different channels in the region of interest.

    Args:
        color_means (dict): A dictionary containing the mean of the ratios of the channels in the colored region.
        tolerance (float, optional): The tolerance for determining the color. Defaults to 0.25.

    Returns:
        str: The determined card color.
    """
    means = color_means

    def is_red():
        # Red cards should have a high r/g and r/b ratio, and a g/b ratio close to 1.
        return (means["r/g"] > 1 + tolerance and means["r/b"] > 1 + tolerance and abs(means["g/b"] - 1) <= tolerance)
    
    def is_green():
        # Green cards should have a low r/g ratio, a r/b ratio close to 1, and a g/b ratio greater than 1.
        # The specific shade of green we want has the red at approximately 65% of the intensity of the green channel and blue channels approximately 55% the intensity of the green channel
        # This means that the r/g ratio should be around 0.6, the r/b ratio should be around 1, and the g/b ratio should be around 2.
        return (means["r/g"] < 0.65 + tolerance and abs(means["r/b"] - 1) <= tolerance and abs(means["g/b"] - 1.8) <= tolerance)
    
    def is_blue():
        # Blue cards should have a low r/g and r/b ratio, and a g/b ratio close to 1.
        # The specific shade of blue we want has the green channel approximately at 80% the intensity of the blue channel
        # This means that the r/g ratio should be low, the r/b ratio should be low, and the g/b ratio should be around 0.8.
        return (means["r/g"] < 1 - tolerance and means["r/b"] < 1 - tolerance and abs(means["g/b"] - 0.8) <= tolerance)
    
    def is_yellow():
        # Yellow cards should have a r/g ratio close to 1, a r/b ratio greater than 1, and a g/b ratio greater than 1.
        # The shade of yellow we want has the red and green channels approximately at the same intensity and the blue channel much lower than the others.
        # This means that the r/g ratio should be around 1, the r/b ratio should be large, and the g/b ratio should be large as well.
        return (abs(means["r/g"] - 1) <= tolerance and means["r/b"] > 1 + tolerance and means["g/b"] > 1 + tolerance)
    
    def is_special():
        # Special cards (like wild cards) might have a black background giving ratios close to 1 for all channels.
        return (abs(means["r/g"] - 1) <= tolerance and abs(means["r/b"] - 1) <= tolerance and abs(means["g/b"] - 1) <= tolerance)
    
    if is_red():
        return "red"
    elif is_green():
        return "green"
    elif is_blue():
        return "blue"
    elif is_yellow():
        return "yellow"
    elif is_special():
        return "special"
    else:
        return "unknown"
    
    
def detect_card_color_from_group_image(group_img, color_detection_tolerance=0.25, plot=False):
    """
    Detect the color of a card from an image of a group of cards.
    This function is useful when the individual cards are not well separated, and we want to determine the color of the cards in a group before separating them.

    Args:
        group_img (numpy.ndarray): The input image of the group of cards. Must be a 3-channel RGB image.
        color_detection_tolerance (float, optional): The tolerance for determining the color. Defaults to 0.25.
        plot (bool, optional): Whether to plot the color features. Defaults to False.

    Returns:
        str: The determined card color for the group.
    """
    # Segment the cards in the group image to create a mask of the card regions !!! This part is what needs to be optimized. --> make separate function segment_cards_from_group
    print("Segmenting cards in the group image...")
    mask = segment_cards(group_img)
    mask_ = clean_mask(mask, close_radius=10)
    
    # Extract blobs from the cleaned mask to identify individual cards
    blobs, centroids_xy, areas = extract_blobs(mask_)
    group_centroids, group_bboxes = blob_bbox(blobs, centroids_xy, areas)
    
    # separate individual cards from the player region using the group bounding boxes and the mask
    print("Extracting images of individual cards from the group image...")
    cards = crop_regions(group_img * np.dstack((mask_, mask_, mask_)), group_bboxes)
    
    print("Determining card color from the extracted card images...")
    for card_idx in cards:
        color = determine_card_color(compute_color_features(cards[card_idx]), color_detection_tolerance)
        print(f"Card {card_idx} is {color}")

    if plot:
        print("Plotting extracted card images...")
        figure, axes = plt.subplots(1, len(cards), figsize=(15, 5))
        if len(cards) == 1:
            axes = [axes]  # Ensure axes is iterable even for a single card
        for ax, card_idx in zip(axes, cards):
            ax.imshow(cards[card_idx])
            ax.set_title(f"Card {card_idx}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    
    return
    