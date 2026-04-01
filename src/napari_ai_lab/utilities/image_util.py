from pathlib import Path

import numpy as np
from skimage import io

# Central list of supported image extensions (lowercase, with leading dot)
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".czi",
}


def collect_all_image_names(image_path):
    """
    Collect all image file paths in a directory using the canonical
    IMAGE_EXTENSIONS set. Returns a sorted list of Path objects.

    Args:
        image_path (str or Path): directory to look for files

    Returns:
        list[Path]: sorted list of image file paths (may be empty)
    """
    image_path = Path(image_path)

    if not image_path.exists() or not image_path.is_dir():
        return []

    image_file_list = [
        p
        for p in image_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_file_list.sort(key=lambda x: x.name.lower())

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


def remove_trivial_axes(axis_types: str, shape: tuple) -> str:
    """
    Remove axis characters corresponding to trivial (singleton) dimensions.

    Args:
        axis_types: Axis string (e.g., "TCZYX")
        shape: Shape tuple corresponding to the axes

    Returns:
        str: Axis string with trivial dimensions removed

    Example:
        >>> remove_trivial_axes("TCZYX", (1, 3, 1, 256, 256))
        "CZYX"
    """
    if not axis_types or len(axis_types) != len(shape):
        return axis_types

    non_trivial_axes = "".join(
        [
            axis
            for axis, size in zip(axis_types, shape, strict=False)
            if size != 1
        ]
    )

    return non_trivial_axes


def get_supported_axes_from_shape(image_shape, supported_axis, image_axis="U"):
    """
    Filter supported axis strings to only those compatible with the given shape.

    Args:
        image_shape (tuple): Shape of the image
        supported_axis (list): List of axis strings to filter

    Returns:
        list: Filtered list of axis strings that are compatible with the shape
    """
    # Deduce axis info from shape
    if image_axis == "U":
        image_axis = get_axis_info_from_shape(image_shape)

    # Determine which axes are present in the deduced axis info
    has_z = "Z" in image_axis
    has_c = "C" in image_axis

    compatible_axes = []
    for axis in supported_axis:
        # Check if this axis string requires Z or C that aren't in the shape
        requires_z = "Z" in axis
        requires_c = "C" in axis

        # Include this axis only if shape supports all required dimensions
        if (not requires_z or has_z) and (not requires_c or has_c):
            compatible_axes.append(axis)

    return compatible_axes


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
    # STEP 1: Find maximum dimensions across all images (robust to axis ordering)
    # =============================================================================

    # Determine presence of Z (3D) and C (color) from axis_infos
    has_3d_images = any(("Z" in ai) for ai in axis_infos)
    has_color_images = any(("C" in ai) for ai in axis_infos)

    def _rows_cols_for_image(image_shape, axis_info):
        """Return (rows, cols) for the image shape using axis_info or fallbacks."""
        # Prefer explicit positions if present
        if "Y" in axis_info:
            row = image_shape[axis_info.index("Y")]
        else:
            row = image_shape[-2] if len(image_shape) >= 2 else image_shape[0]

        if "X" in axis_info:
            col = image_shape[axis_info.index("X")]
        else:
            col = image_shape[-1] if len(image_shape) >= 1 else image_shape[0]

        return row, col

    def _depth_for_image(image_shape, axis_info):
        """Return depth (Z) size for the image or 0 if none."""
        if "Z" in axis_info:
            return image_shape[axis_info.index("Z")]
        return 0

    # Compute maximum rows/cols/depth across all images using per-image axis info
    max_rows = max(
        _rows_cols_for_image(img.shape, ai)[0]
        for img, ai in zip(images, axis_infos, strict=False)
    )
    max_cols = max(
        _rows_cols_for_image(img.shape, ai)[1]
        for img, ai in zip(images, axis_infos, strict=False)
    )
    max_depth = 0
    if has_3d_images:
        max_depth = max(
            _depth_for_image(img.shape, ai)
            for img, ai in zip(images, axis_infos, strict=False)
            if "Z" in ai
        )

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
                axis_info = "ZYXC"

        sh = image.shape
        ndim = len(sh)

        # Build a pad_width list sized to the image ndim
        pad_width = [(0, 0)] * ndim

        # Determine indices for Y, X, and Z axes (fall back to last axes if not present)
        if "Y" in axis_info:
            y_idx = axis_info.index("Y")
        else:
            y_idx = ndim - 2 if ndim >= 2 else 0

        if "X" in axis_info:
            x_idx = axis_info.index("X")
        else:
            x_idx = ndim - 1 if ndim >= 1 else 0

        if max_depth > 0:
            # If the image doesn't have Z but global max_depth > 0, we already
            # expanded it earlier for has_3d_images, so fall back to first dim.
            z_idx = axis_info.index("Z") if "Z" in axis_info else 0

        # Compute padding amounts from mapped indices
        pad_rows = max_rows - sh[y_idx]
        if pad_rows < 0:
            pad_rows = 0
        pad_width[y_idx] = (0, pad_rows)

        pad_cols = max_cols - sh[x_idx]
        if pad_cols < 0:
            pad_cols = 0
        pad_width[x_idx] = (0, pad_cols)

        if max_depth > 0:
            pad_depth = max_depth - sh[z_idx]
            if pad_depth < 0:
                pad_depth = 0
            pad_width[z_idx] = (0, pad_depth)

        # Handle channel clipping (keep first 3 channels if more are present)
        if "C" in axis_info:
            c_idx = axis_info.index("C")
            if sh[c_idx] > 3:
                # build slicer to clip channels
                slicer = [slice(None)] * ndim
                slicer[c_idx] = slice(0, 3)
                image = image[tuple(slicer)]
                sh = image.shape
                # Keep pad_width length consistent; no change required

        # Apply padding with the computed pad_width
        padded_image = np.pad(
            image,
            pad_width,
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


def get_current_slice_indices(
    current_step: tuple, selected_axis: str, ignore_channel: bool = False
):
    """Compute indices for the current slice based on the selected axis.

    Args:
        current_step: The napari viewer dims.current_step tuple.
        selected_axis: Axis mode string, e.g. "YX" or "ZYX".
        ignore_channel: If True, remove 'C' from selected_axis before processing.

    Returns:
        tuple: A tuple of indices/slices to extract the current 2D/3D region.
    """
    # Remove channel dimension if requested
    if ignore_channel and "C" in selected_axis:
        selected_axis = selected_axis.replace("C", "")

    if selected_axis == "YX":
        return current_step[:-2] + (slice(None), slice(None))
    elif selected_axis == "ZYX":
        return current_step[:-3] + (
            slice(None),
            slice(None),
            slice(None),
        )
    if selected_axis == "YXC":
        return current_step[:-3] + (slice(None), slice(None), slice(None))
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


def create_empty_instance_image(image_shape, dtype=np.uint16):
    """
    Create an empty instance-label image matching the provided image_shape.

    For color images (axis info 'YXC') we return a 2D YX zeros array. For
    other shapes we return zeros with the same shape.

    Args:
        image_shape (tuple): Shape of the reference image
        dtype: numpy dtype for the returned array (default: np.uint16)

    Returns:
        numpy.ndarray: Zero-filled array suitable for annotations/predictions
    """
    axis_info = get_axis_info_from_shape(image_shape)

    if axis_info == "YXC":
        return np.zeros(image_shape[:2], dtype=dtype)
    return np.zeros(image_shape, dtype=dtype)
