import os
import matplotlib.pyplot as plt
import numpy as np

def load_all_images(image_dir):
    """
    Load all images in the specified directory.
    
    Args:
        image_dir (str): The directory containing the images.

    Returns:
        tuple: A tuple containing the dictionary of loaded images and a list of image IDs.
    """
    images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            images[filename] = plt.imread(image_path)
    
    image_ids = [filename[:-4] for filename in list(images.keys())]  # Create a list of image IDs by removing the .jpg extension
            
    return images, image_ids

def select_image_subset(images, subset_id):
    """ Selects a subset of images based on the last digit of their IDs.
    Args:
        images (dict): A dictionary of images with keys as filenames (without extension) and values as image data. Just like the one returned by load_all_images.
        subset_id (int): The digit to filter the image IDs by:
            0 -> White background, separated cards
            1 -> White background, grouped cards
            2 -> Colored background, separated cards
            3 -> Colored background, grouped cards
            4 -> All white background images (both separated and grouped)
            5 -> All colored background images (both separated and grouped)
            6 -> All images (both white and colored backgrounds, both separated and grouped)
    Returns:
        dict: A dictionary of selected images with keys as filenames (without extension) and values as image data.
    """
    image_ids = list(images.keys())
    
    if subset_id in [0,1]:
        # Select images with white background (IDs ending in 0 or 1)
        if subset_id == 0:
            key = "7"
        elif subset_id == 1:
            key = "8"
        selected_images = {image_id: images[image_id + ".jpg"] for image_id in image_ids if image_id[-3] == key}
        print(f"Selected images: {list(selected_images.keys())}")
    elif subset_id in [2,3]:
        if subset_id == 2:
            key = "9"
            subgroup = ("0", "1", "2")
        elif subset_id == 3:
            key = "9"
            subgroup = ("6", "7", "8")
        # Select images with green background (IDs ending in 2 or 3)
        selected_images = {image_id: images[image_id + ".jpg"] for image_id in image_ids if (image_id[-3] == key and image_id[-2] in subgroup)}
        print(f"Selected images: {list(selected_images.keys())}")
    elif subset_id == 4:
        # Select all white background images (IDs ending in 0 or 1)
        selected_images = {image_id: images[image_id + ".jpg"] for image_id in image_ids if image_id[-3] in ["7", "8"]}
        print(f"Selected images: {list(selected_images.keys())}")
    elif subset_id == 5:
        # Select all colored background images (IDs ending in 2 or 3)
        selected_images = {image_id: images[image_id + ".jpg"] for image_id in image_ids if image_id[-3] in ["9"]}
        print(f"Selected images: {list(selected_images.keys())}")
    elif subset_id == 6:
        # Select all images
        selected_images = images
        print(f"Selected images: {list(selected_images.keys())}")
    
    return selected_images

def load_image_subset(image_dir, subset_id):
    """ Loads a subset of images based on the last digit of their IDs.
    Args:
        image_dir (str): The directory containing the images.
        subset_id (int): The digit to filter the image IDs by:
            0 -> White background, separated cards
            1 -> White background, grouped cards
            2 -> Colored background, separated cards
            3 -> Colored background, grouped cards
            4 -> All white background images (both separated and grouped)
            5 -> All colored background images (both separated and grouped)
            6 -> All images (both white and colored backgrounds, both separated and grouped)
    Returns:
        tuple: A tuple containing the dictionary of loaded images and a list of image IDs.
    """
    
    all_image_ids = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            all_image_ids.append(filename[:-4])  # Create a list of image IDs by removing the .jpg extension
    
    
    if subset_id in [0,1]:
        # Select images with white background (IDs ending in 0 or 1)
        if subset_id == 0:
            key = "7"
        elif subset_id == 1:
            key = "8"
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] == key]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id in [2,3]:
        if subset_id == 2:
            key = "9"
            subgroup = ("0", "1", "2")
        elif subset_id == 3:
            key = "9"
            subgroup = ("6", "7", "8")
        # Select images with green background (IDs ending in 2 or 3)
        selected_images_ids = [image_id for image_id in all_image_ids if (image_id[-3] == key and image_id[-2] in subgroup)]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 4:
        # Select all white background images (IDs ending in 0 or 1)
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] in ["7", "8"]]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 5:
        # Select all colored background images (IDs ending in 2 or 3)
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] in ["9"]]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 6:
        # Select all images
        selected_images_ids = all_image_ids
        print(f"Selected images: {selected_images_ids}")

    selected_images = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") and filename[:-4] in selected_images_ids:
            image_path = os.path.join(image_dir, filename)
            selected_images[filename] = plt.imread(image_path)
        
        
    return selected_images, selected_images_ids

def remove_background(image, background_image):
    """Removes the background from the input image by computing the absolute difference with the background image.
    Args:
        image (numpy.ndarray): The input image.
        background_image (numpy.ndarray): The background image.

    Returns:
        numpy.ndarray: The image with the background removed.
    """
    
    # check if the input image and background image have the same pixel values range (0-1)
    if image.max() > 1.0:
        print("Normalizing input image to the range [0, 1].")
        image = image / 255.0
    elif background_image.max() > 1.0:
        print("Normalizing background image to the range [0, 1].")
        background_image = background_image / 255.0
    
    
    return np.abs(image - background_image)

def preprocess_background(image, bg_type, reference_bg=None):
    """
    Conditional background preprocessing.

    Args:
        image (np.ndarray)       : RGB image, uint8 [0,255] or float [0,1].
        bg_type (str)            : 'white' or 'noisy', from detect_background_type().
        reference_bg (np.ndarray): reference background image (only used when
                                   bg_type == 'noisy').

    Returns:
        np.ndarray, float [0, 1].
    """
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    if bg_type == "white":
        return img
    if bg_type == "noisy":
        if reference_bg is None:
            raise ValueError("reference_bg is required for noisy backgrounds.")
        return remove_background(img, reference_bg)
    raise ValueError(f"Unknown bg_type: {bg_type!r}. Expected 'white' or 'noisy'.")
