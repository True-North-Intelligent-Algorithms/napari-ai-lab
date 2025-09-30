import numpy as np
from skimage import io
from tnia.io.io_helper import collect_all_image_names


def load_images_from_directory(directory):
    """
    Load all images from a directory into a list.

    This function will eventually be expanded to handle axis info detection
    and other image metadata processing.

    Args:
        directory (str): Path to directory containing images

    Returns:
        tuple: (images_list, image_paths_list) or (None, None) if no images found
    """
    # Collect all image file paths
    image_names = collect_all_image_names(directory)
    if len(image_names) == 0:
        return None, None

    print(f"Loading {len(image_names)} images...")

    # Load each image into a list
    images = []
    axis_infos = []
    successful_paths = []

    for image_path in image_names:
        try:
            image = io.imread(image_path)
            axis_info = get_axis_info(image)
            images.append(image)
            axis_infos.append(axis_info)
            successful_paths.append(image_path)
            print(f"Loaded: {image_path} (axis: {axis_info})")
            # TODO: Check file for axis info metadata or override deduced info
        except (OSError, ValueError, ImportError) as e:
            print(f"Failed to load {image_path}: {e}")
            continue

    if len(images) == 0:
        return None, None, None

    return images, axis_infos, successful_paths


def get_axis_info(image):
    """
    Returns the axis string for an image based on its dimensions.
    Uses Python array ordering convention (YX, ZYX, YXC).

    Args:
        image (numpy.ndarray): Input image array

    Returns:
        str: Axis string describing the dimension order ('U' = unknown)
    """
    if len(image.shape) == 2:
        return "YX"
    if len(image.shape) == 3:
        # Heuristic: if the last dimension is 3 or 4, assume it's color channels
        if image.shape[2] in [3, 4]:
            return "YXC"
        else:
            return "ZYX"
    else:
        # Return string of 'U' (unknown) same length as number of dimensions
        return "U" * len(image.shape)


def pad_to_largest(
    images, axis_infos, force8bit=False, normalize_per_channel=False
):
    """
    Pads a list of images to the largest dimensions in the list.

    This is useful for displaying images of different sizes as a sequence in Napari

    Args:
        images (list): list of images to pad
        axis_infos (list): list of axis info strings corresponding to each image
        force8bit (bool): whether to normalize the images to 8 bit
        normalize_per_channel (bool): whether to normalize the images per channel

        Returns:
        numpy.ndarray: The padded images
    """

    # TODO: this is actually a pretty complicated function, not only do we pad but
    # we also normalize the images to 8 bit, and we also convert to rgb if the image is not 3 channel
    # will need continued work and refactoring, as we are essentially doing multi-time, multi-channel, multi-format
    # conversion for display.

    # =============================================================================
    # STEP 1: Find maximum dimensions across all images
    # =============================================================================
    max_rows = max(image.shape[0] for image in images)
    max_cols = max(image.shape[1] for image in images)

    # Check if we have any color images (YXC) - if so, convert grayscale to RGB
    has_color_images = any(axis_info == "YXC" for axis_info in axis_infos)

    # =============================================================================
    # STEP 2: Process each image (pad + convert channels + normalize)
    # =============================================================================
    padded_images = []

    for image, axis_info in zip(images, axis_infos, strict=False):
        # Calculate the padding for each dimension
        pad_rows = max_rows - image.shape[0]
        pad_cols = max_cols - image.shape[1]

        # Convert grayscale to RGB if we have color images in the set
        if has_color_images and axis_info == "YX":
            # Convert 2D grayscale to 3D RGB by adding channel dimension and repeating
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
            axis_info = "YXC"  # Update axis_info after conversion

        # Handle different axis types
        if axis_info == "YXC":
            # 3D image with channels - we occasionally hit rgba images, just use the first 3 channels
            image = image[:, :, :3]
            # Pad Y and X dimensions, leave channels unchanged
            padded_image = np.pad(
                image,
                ((0, pad_rows), (0, pad_cols), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif axis_info == "YX":
            # 2D grayscale image - pad Y and X dimensions
            padded_image = np.pad(
                image,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )
        else:
            # For other axis types, treat as 2D for now (fallback)
            padded_image = np.pad(
                image,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )

        # TODO: Implement remove_alpha_channel function
        # if len(padded_image.shape) > 2 and padded_image.shape[2] != 3:
        #     padded_image = remove_alpha_channel(padded_image)

        if force8bit:

            if (len(padded_image.shape) > 2) and normalize_per_channel:
                padded_image = normalize_per_channel(padded_image)
            else:
                min_ = np.min(padded_image)
                max_ = np.max(padded_image)
                padded_image = (
                    (padded_image - min_) / (max_ - min_) * 255
                ).astype(np.uint8)

        padded_images.append(padded_image)

    # =============================================================================
    # STEP 3: Stack all processed images into final result
    # =============================================================================

    # BN was toying with the idea of displaying 3 channel and 1 channel images together but it is a bit messy, so commented out for now
    """
    shapes = [image.shape for image in padded_images]
    if len(set(shapes)) > 1:
        for i in range(len(padded_images)):
            if len(padded_images[i].shape)==2:
                padded_images[i] = padded_images[i][:,:,np.newaxis]
                padded_images[i] = multi_channel_to_rgb(padded_images[i])
    """

    # Stack the padded images along a new third dimension
    result = np.array(padded_images)

    if force8bit:
        result = result.astype(np.uint8)

    return result
