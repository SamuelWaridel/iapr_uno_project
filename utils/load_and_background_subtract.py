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

def load_image_subset(image_dir, subset_id, return_features=False):
    """ Loads a subset of images based on the last digit of their IDs.
    Args:
        image_dir (str): The directory containing the images.
        subset_id (int): The digit to filter the image IDs by:
            0 -> Illustrative subset, one image per category (IDs: "L1000772","L1000836","L1000910", "L1000973")
            1 -> White background, separated cards
            2 -> White background, grouped cards
            3 -> Colored background, separated cards
            4 -> Colored background, grouped cards
            5 -> All white background images (both separated and grouped)
            6 -> All colored background images (both separated and grouped)
            7 -> All images (both white and colored backgrounds, both separated and grouped)
        return_features (bool): If True, return a dictionary of features instead of raw images. The dictionary will have the format:
            {image_id: 
                {"image": np.ndarray,  # The loaded image as a NumPy array
                # ... Additional features can be added here in the future
                }
            }
    Returns:
        tuple: A tuple containing the dictionary of loaded images and a list of image IDs.
    """
    
    all_image_ids = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            all_image_ids.append(filename[:-4])  # Create a list of image IDs by removing the .jpg extension
    
    if subset_id == 0:
        # Illustrative subset, one image per category (IDs: "L1000772","L1000836","L1000910", "L1000973")
        selected_images_ids = ["L1000772", "L1000836", "L1000910", "L1000973"]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id in [1,2]:
        # Select images with white background (IDs ending in 0 or 1)
        if subset_id == 1:
            key = "7"
        elif subset_id == 2:
            key = "8"
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] == key]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id in [3,4]:
        if subset_id == 3:
            key = "9"
            subgroup = ("0", "1", "2")
        elif subset_id == 4:
            key = "9"
            subgroup = ("6", "7", "8")
        # Select images with green background (IDs ending in 2 or 3)
        selected_images_ids = [image_id for image_id in all_image_ids if (image_id[-3] == key and image_id[-2] in subgroup)]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 5:
        # Select all white background images (IDs ending in 0 or 1)
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] in ["7", "8"]]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 6:
        # Select all colored background images (IDs ending in 2 or 3)
        selected_images_ids = [image_id for image_id in all_image_ids if image_id[-3] in ["9"]]
        print(f"Selected images: {selected_images_ids}")
    elif subset_id == 7:
        # Select all images
        selected_images_ids = all_image_ids
        print(f"Selected images: {selected_images_ids}")

    selected_images = {
        id: plt.imread(os.path.join(image_dir, id + ".jpg"))
        for id in selected_images_ids
    }
    
    if return_features:
        features = {img_id: {"image": selected_images[img_id]} for img_id in selected_images_ids}
        return features
    else:
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
