from pathlib import Path

import numpy as np
from skimage import io


def collect_all_image_names(image_path, extensions=None):
    """
    Collects all image names

    Args:
        image_path (Path): directory to look for files
        extensions (list): list of extensions to look for

    Returns:
        list: list of image file names
    """
    if extensions is None:
        extensions = ["jpg", "jpeg", "tif", "tiff", "png"]

    image_path = Path(image_path)

    image_file_list = []

    for extension in extensions:
        image_file_list = image_file_list + list(
            image_path.glob("*." + extension)
        )

    return image_file_list


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
    return get_axis_info_from_shape(image.shape)


def get_axis_info_from_shape(image_shape):
    if len(image_shape) == 2:
        return "YX"
    if len(image_shape) == 3:
        # Heuristic: if the last dimension is 3 or 4, assume it's color channels
        if image_shape[2] in [3, 4]:
            return "YXC"
        else:
            return "ZYX"
    else:
        # Return string of 'U' (unknown) same length as number of dimensions
        return "U" * len(image_shape)


def get_ndim(shape):
    """
    Determine spatial dimensionality from image shape.

    Args:
        shape (tuple): Image shape

    Returns:
        int: 2 for 2D images, 3 for 3D images
    """
    if len(shape) == 2:
        return 2
    if len(shape) == 3:
        return 2 if shape[-1] < 5 else 3
    return len(shape)


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
    # Find max Y and X dimensions (always present)
    max_rows = max(image.shape[-2] for image in images)  # Y dimension
    max_cols = max(image.shape[-1] for image in images)  # X dimension

    # Find max Z dimension for 3D images (ZYX)
    max_depth = 0
    has_3d_images = any(axis_info == "ZYX" for axis_info in axis_infos)
    if has_3d_images:
        max_depth = max(
            image.shape[0]
            for image, axis_info in zip(images, axis_infos, strict=False)
            if axis_info == "ZYX"
        )

    # Check if we have any color images (YXC) - if so, convert grayscale to RGB
    has_color_images = any(axis_info == "YXC" for axis_info in axis_infos)

    # =============================================================================
    # STEP 2: Process each image (pad + convert channels + normalize)
    # =============================================================================
    padded_images = []

    for image, axis_info in zip(images, axis_infos, strict=False):

        # Convert grayscale to RGB if we have color images in the set
        if has_color_images and axis_info == "YX":
            # Convert 2D grayscale to 3D RGB by adding channel dimension and repeating
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
            axis_info = "YXC"  # Update axis_info after conversion

        # Convert 2D images to 3D if we have 3D images in the set
        if has_3d_images and axis_info in ["YX", "YXC"]:
            # Add Z dimension at the beginning
            image = np.expand_dims(image, axis=0)
            if axis_info == "YX":
                axis_info = "ZYX"
            elif axis_info == "YXC":
                axis_info = "ZYXC"  # This would be 4D, handle carefully

        # Calculate padding for each dimension based on axis type
        if axis_info == "YXC":
            # 3D image with channels: pad Y,X dimensions
            pad_rows = max_rows - image.shape[0]  # Y
            pad_cols = max_cols - image.shape[1]  # X
            # Clip to first 3 channels if RGBA
            image = image[:, :, :3]
            padded_image = np.pad(
                image,
                ((0, pad_rows), (0, pad_cols), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        elif axis_info == "YX":
            # 2D grayscale image: pad Y,X dimensions
            pad_rows = max_rows - image.shape[0]  # Y
            pad_cols = max_cols - image.shape[1]  # X
            padded_image = np.pad(
                image,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )
        elif axis_info == "ZYX":
            # 3D spatial image: pad Z,Y,X dimensions
            pad_depth = max_depth - image.shape[0]  # Z
            pad_rows = max_rows - image.shape[1]  # Y
            pad_cols = max_cols - image.shape[2]  # X
            padded_image = np.pad(
                image,
                ((0, pad_depth), (0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=0,
            )
        elif axis_info == "ZYXC":
            # 4D image with Z,Y,X,C: pad Z,Y,X dimensions
            pad_depth = max_depth - image.shape[0]  # Z
            pad_rows = max_rows - image.shape[1]  # Y
            pad_cols = max_cols - image.shape[2]  # X
            # Clip to first 3 channels if RGBA
            image = image[:, :, :, :3]
            padded_image = np.pad(
                image,
                ((0, pad_depth), (0, pad_rows), (0, pad_cols), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            # Fallback: treat as 2D and pad Y,X dimensions
            pad_rows = max_rows - image.shape[-2]  # Y (second to last)
            pad_cols = max_cols - image.shape[-1]  # X (last)
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


def get_current_slice_indices(current_step: tuple, selected_axis: str):
    """Compute indices for the current slice based on the selected axis.

    Args:
        current_step: The napari viewer dims.current_step tuple.
        selected_axis: Axis mode string, e.g. "YX" or "ZYX".

    Returns:
        tuple: A tuple of indices/slices to extract the current 2D/3D region.
    """
    if selected_axis == "YX":
        return current_step[:-2] + (slice(None), slice(None))
    elif selected_axis == "ZYX":
        return current_step[:-3] + (
            slice(None),
            slice(None),
            slice(None),
        )
    else:
        # Default to 2D YX slice
        return current_step[:-2] + (slice(None), slice(None))


def create_artifact_name(
    artifact_base: str, step: tuple, selected_axis: str
) -> str:
    """Create an artifact name based on current dims step and selected axis.

    The name is built from the base plus non-spatial dimension indices from
    the current viewer step. Spatial dims are determined from the selected_axis:
    - endswith("ZYX"): 3 spatial dims (Z, Y, X)
    - endswith("YX"): 2 spatial dims (Y, X)
    - default: 2 spatial dims

    Args:
        artifact_base: Base name (e.g., image stem or artifact id)
        step: napari viewer dims.current_step tuple
        selected_axis: Axis string like "YX", "ZYX", "YXC", etc.

    Returns:
        str: Artifact name with non-spatial dimension indices appended.
    """
    if selected_axis.endswith("ZYX"):
        spatial = 3
    elif selected_axis.endswith("YX"):
        spatial = 2
    else:
        spatial = 2

    non_spatial_dims = step[:-spatial]

    if len(non_spatial_dims) == 0:
        return artifact_base

    suffix = "_".join(str(d) for d in non_spatial_dims)
    return f"{artifact_base}_{suffix}"
