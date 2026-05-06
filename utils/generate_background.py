import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from skimage.morphology import opening, closing, disk, remove_small_objects
import os

PATH_TO_BACKGROUND_IMAGE = 'images/background_image.png'

def generate_background_image(image_subset, simple=True, save_background_image=False, path_to_save=PATH_TO_BACKGROUND_IMAGE):
    """Generates a background image for a subset of images by combining the pixel values across the subset.
    Args:
        image_subset (list): A list of RGB images pertaining to a given category.
        simple (bool): If True, uses a simple median filter to generate the background image. If False, uses a more complex method that identifies ambiguous areas and fills them more carefully.
        save_background_image (bool): If True, saves the generated background image as an png.
        path_to_save (str): The file path to save the generated background image if save_background_image is True.
    Returns:
        tuple: A 3-channel background image representing the average background across the subset and a boolean mask indicating ambiguous areas.
    """
    background_hues = []
    background_sats = []
    background_vals = []
    
    for image in image_subset: # Loop through each image in the subset. Takes a while to run. (6.5 minutes for the colored background subset)
        hsv_img = rgb2hsv(image / 255.0)
        background_hues.append(hsv_img[:,:,0])
        background_sats.append(hsv_img[:,:,1])
        background_vals.append(hsv_img[:,:,2])
    
    if simple:
        # Compute the median across the subset to get a representative background color.
        background_hue = np.median(np.array(background_hues), axis=0)
        background_sat = np.median(np.array(background_sats), axis=0)
        background_val = np.median(np.array(background_vals), axis=0)
        
        ambiguous = np.zeros_like(background_hue, dtype=bool)  # No ambiguous areas in simple mode
        
    elif not simple:
        # Define the ambiguous areas of the background based on the standard deviation of the saturation channel across the subset.
        sats_stdevs = np.std(background_sats, axis=0)
        ambiguous = sats_stdevs > np.quantile(sats_stdevs.ravel(), 0.95)
        
        #Use morphological operations to clean up the ambiguous mask.
        close_radius=5
        open_radius=5
        min_blob_size=100
        # Remove thin noise
        opened = opening(sats_stdevs > np.quantile(sats_stdevs.ravel(), 0.95), disk(open_radius))
        # Close gaps
        closed = closing(opened, disk(close_radius))
        # Discard blobs that are too small to be part of the ambiguous region
        ambiguous = remove_small_objects(closed, min_size=min_blob_size)

        
        # Compute the median across the non-ambiguous areas to get a representative background color.
        background_hue = np.median(np.array(background_hues), axis=0)
        background_sat = np.median(np.array(background_sats), axis=0)
        background_val = np.median(np.array(background_vals), axis=0)
        
        # Set the ambiguous areas to zero so that they don't affect the image when substracting the background.
        # That way the maximal information of the image is preserved in the ambiguous areas, which are mostly the areas where the cards are located.
        background_hue[ambiguous] = 0.0  
        background_sat[ambiguous] = 0.0  
        background_val[ambiguous] = 0.0  
    
    # Combine the background channels into a single RGB image (after converting back from HSV).
    background_rgb = hsv2rgb(np.stack([background_hue, background_sat, background_val], axis=-1))
    
    # Save the background image as an image (optional, for visualization purposes).
    if save_background_image:
        plt.imsave(path_to_save, background_rgb)

    return background_rgb, ambiguous


def load_background_image(path=PATH_TO_BACKGROUND_IMAGE):
    """Loads a previously saved background image.
    Args:
        path (str): The file path to the background image.
    Returns:
        np.ndarray: A 3-channel background image.
    """
    background_rgb = plt.imread(path)
    
    # If the image has an alpha channel, discard it to keep only the RGB channels.
    if background_rgb.shape[-1] == 4:
        background_rgb = background_rgb[:,:, :3]
        
    return background_rgb

def generate_background_image_from_crops(list_image_filepaths = None, save_path=None):
    
    """
    Generate a background image using crops of known images.
    Was run in report.ipynb using this list of filepaths: 
        list_image_filepaths = [
        "data/train_images/L1000965.jpg",
        "data/train_images/L1000964.jpg",
        "data/train_images/L1000972.jpg",
        "data/train_images/L1000968.jpg",
        "images/colored_background_simple.png"
        ]

    Arguments:
        list_image_filepaths (list of str): List of file paths to the images to be used for generating the background image. 
        The order of the images should be: [no_card_on_top, no_card_on_bottom, no_card_on_right, no_card_on_left, center_image].
        save_path (str, optional): If provided, the generated background image will be saved to this path.
        
    Returns:
        numpy.ndarray: The generated background image.
    """

    if list_image_filepaths is None:
        raise ValueError("No image file paths provided. Please provide a list of image file paths to generate the background image.")
    
    images = []
       
    for filepath in list_image_filepaths:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}. Please check the file paths.")
        else:
            print(f"Loading image: {filepath}")
            new_image = plt.imread(filepath)
            if new_image is None:   
                raise ValueError(f"Could not load image: {filepath}. Please check the file paths and ensure the images are in a supported format.")
            if new_image.shape[2] != 3:
                new_image = new_image[:, :, :3]  # Keep only the first 3 channels if there are more (e.g., RGBA)
            images.append(new_image)
    
    shapex, shapey, _ = images[0].shape
    for img in images:
        if img is None:
            raise ValueError("One of the images could not be loaded. Please check the file paths.")
        # Normalize all images to have values in the range [0,1]
        if not np.all((img >= 0) & (img <= 1)):
            img = img / 255.0
        if shapex != img.shape[0] or shapey != img.shape[1]:
            raise ValueError("All images must have the same dimensions. Please check the file paths and image sizes.")

    # Define the coordinates for the masks
    right = 2450
    left = 1650
    top = 800
    bottom = 1700

    X,Y,C = images[0].shape  # Assuming no_bottom is the second image in the list
    
    # Create masks for each region
    mask_no_card_on_top = np.ones((X, Y, C), dtype=bool)  # Start with all True (white)
    mask_no_card_on_top[top:, :] = False  # Set top part to False (black)

    mask_no_card_on_bottom = np.ones((X, Y, C), dtype=bool)  # Start with all True (white)
    mask_no_card_on_bottom[:bottom, :] = False  # Set top part to False (black)

    mask_no_card_on_right = np.ones((X, Y, C), dtype=bool)  # Start with all True (white)
    mask_no_card_on_right[:, :right] = False  # Set top part to False (black)

    mask_no_card_on_left = np.ones((X, Y, C), dtype=bool)  # Start with all True (white)
    mask_no_card_on_left[:, left:] = False  # Set top part to False (black)

    mask_center = mask_no_card_on_top | mask_no_card_on_bottom | mask_no_card_on_right | mask_no_card_on_left
    
    # Combine the images using the masks

    outer_ring = images[0] * mask_no_card_on_top + images[1] * mask_no_card_on_bottom + images[2] * mask_no_card_on_right * (~mask_no_card_on_top & ~mask_no_card_on_bottom) + images[3] * mask_no_card_on_left * (~mask_no_card_on_top & ~mask_no_card_on_bottom)

    mask_center = ~mask_center

    total_image = outer_ring / 255.0 + images[4] * mask_center
    
    if save_path is not None:
        plt.imsave(save_path, total_image)
    
    return total_image
