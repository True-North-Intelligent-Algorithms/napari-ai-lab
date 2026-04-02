"""Deep learning utility functions."""

import os

import numpy as np
from skimage import io
from tqdm import tqdm


def normalize_image(image, p_low=None, p_high=None):
    """
    Normalize image using explicit low and high values or percentile normalization.

    This normalizes the image by scaling values between p_low and p_high to [0, 1].
    If p_low and p_high are not provided, they are computed from the image itself.

    Args:
        image (numpy.ndarray): Input image
        p_low (float, optional): Low value for normalization. If None, computed as 1st percentile.
        p_high (float, optional): High value for normalization. If None, computed as 99th percentile.

    Returns:
        numpy.ndarray: Normalized image in range [0, 1]
    """
    image_float = image.astype(np.float32)

    # Compute percentiles if not provided
    if p_low is None or p_high is None:
        computed_low, computed_high = np.percentile(image_float, [1, 99])
        p_low = computed_low if p_low is None else p_low
        p_high = computed_high if p_high is None else p_high

    # Normalize using provided or computed values
    if p_high > p_low:
        image_norm = (image_float - p_low) / (p_high - p_low)
        image_norm = np.clip(image_norm, 0, 1)
    else:
        image_norm = image_float

    return image_norm


def normalize_percentile(image, percentile_low=1, percentile_high=99):
    """
    Compute percentile values from image for use in global normalization.

    This is useful for computing global statistics from a full image/label
    that can then be applied to individual patches for consistent normalization.

    Args:
        image (numpy.ndarray): Input image to compute percentiles from
        percentile_low (float): Lower percentile (default: 1)
        percentile_high (float): Upper percentile (default: 99)

    Returns:
        tuple: (p_low, p_high) percentile values
    """
    image_float = image.astype(np.float32)
    p_low, p_high = np.percentile(
        image_float, [percentile_low, percentile_high]
    )
    return p_low, p_high


def collect_training_data(
    data_path: str,
    sub_sample: int = 1,
    downsample: bool = False,
    pmin: float = 3,
    pmax: float = 99.8,
    normalize_input: bool = True,
    normalize_truth: bool = False,
    training_multiple: int = 1,
    add_trivial_channel: bool = True,
    relabel: bool = False,
) -> tuple[list, list]:
    """
    Collect pre-generated patch pairs from a patches directory.

    Reads all .tif pairs from ``data_path/input0/`` and
    ``data_path/ground_truth0/``, optionally normalizing and reshaping
    each pair before returning them as lists ready for model training.

    Args:
        data_path: Path to the patches directory (contains input0/ and ground_truth0/).
        sub_sample: Take every Nth file (1 = all files, 2 = every other, etc.).
        downsample: If True, spatially downsample by 2× on Y and X axes.
        pmin: Lower percentile for input normalization.
        pmax: Upper percentile for input normalization.
        normalize_input: If True, percentile-normalize input patches.
        normalize_truth: If True, percentile-normalize ground-truth patches.
        training_multiple: Crop spatial dims to the nearest multiple of this value.
            Useful for networks that require dimensions divisible by 16 or 32.
        add_trivial_channel: If True, append a channel axis (e.g. for CARE/StarDist).
        relabel: If True, re-label ground-truth with connected-component labeling.

    Returns:
        (X, Y): lists of input and ground-truth arrays.
    """
    from skimage.measure import label as skimage_label

    input_dir = os.path.join(data_path, "input0")
    truth_dir = os.path.join(data_path, "ground_truth0")

    input_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".tif")
    )[::sub_sample]
    truth_files = sorted(
        f for f in os.listdir(truth_dir) if f.endswith(".tif")
    )[::sub_sample]

    X, Y = [], []

    for input_name, truth_name in tqdm(
        zip(input_files, truth_files, strict=False)
    ):
        x = io.imread(os.path.join(input_dir, input_name), plugin="tifffile")
        y = io.imread(os.path.join(truth_dir, truth_name), plugin="tifffile")

        if downsample:
            x = x[..., ::2, ::2]
            y = y[..., ::2, ::2]

        if training_multiple > 1:
            shape = x.shape
            slices = tuple(
                slice(0, (s // training_multiple) * training_multiple)
                for s in shape[-x.ndim :]
            )
            x = x[slices]
            y = y[slices]

        if normalize_input:
            x = normalize_image(
                x, *np.percentile(x.astype(np.float32), [pmin, pmax])
            )

        if normalize_truth:
            y = normalize_image(
                y, *np.percentile(y.astype(np.float32), [pmin, pmax])
            )

        if add_trivial_channel:
            x = x[..., np.newaxis]

        if relabel:
            y = skimage_label(y)

        X.append(x)
        Y.append(y)

    return X, Y


def divide_training_data(
    X: list,
    Y: list,
    val_size: int = 2,
) -> tuple[list, list, list, list]:
    """
    Split collected training data into training and validation sets.

    Takes the last *val_size* items as validation and the rest as training.

    Args:
        X: List of input arrays.
        Y: List of ground-truth arrays.
        val_size: Number of samples to reserve for validation.

    Returns:
        (X_train, Y_train, X_val, Y_val)
    """
    X_train = X[:-val_size]
    Y_train = Y[:-val_size]
    X_val = X[-val_size:]
    Y_val = Y[-val_size:]
    return X_train, Y_train, X_val, Y_val
