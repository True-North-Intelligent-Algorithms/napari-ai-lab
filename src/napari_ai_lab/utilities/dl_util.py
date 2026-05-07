"""Deep learning utility functions."""

import os

import numpy as np
from skimage import io
from tqdm import tqdm


def normalize_intensity(image, intensity_low, intensity_high):
    """
    Normalize image to [0, 1] using explicit intensity bounds.

    Args:
        image (numpy.ndarray): Input image.
        intensity_low (float): Intensity value that maps to 0.
        intensity_high (float): Intensity value that maps to 1.

    Returns:
        numpy.ndarray: float32 array clipped to [0, 1].
    """
    image_float = image.astype(np.float32)
    if intensity_high > intensity_low:
        image_float = (image_float - intensity_low) / (
            intensity_high - intensity_low
        )
        image_float = np.clip(image_float, 0, 1)
    return image_float


def compute_percentiles(image, percentile_low=1, percentile_high=99):
    """
    Compute percentile intensity values from an image.

    Args:
        image (numpy.ndarray): Input image.
        percentile_low (float): Lower percentile (default: 1).
        percentile_high (float): Upper percentile (default: 99).

    Returns:
        tuple: (low, high) intensity values.
    """
    image_float = image.astype(np.float32)
    return tuple(np.percentile(image_float, [percentile_low, percentile_high]))


def normalize_percentile(image, percentile_low=1, percentile_high=99):
    """
    Normalize image to [0, 1] using percentile-based intensity bounds.

    Computes percentile values from the image, then normalizes.

    Args:
        image (numpy.ndarray): Input image.
        percentile_low (float): Lower percentile (default: 1).
        percentile_high (float): Upper percentile (default: 99).

    Returns:
        numpy.ndarray: float32 array clipped to [0, 1].
    """
    low, high = compute_percentiles(image, percentile_low, percentile_high)
    return normalize_intensity(image, low, high)


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
            x = normalize_percentile(x, pmin, pmax)

        if normalize_truth:
            y = normalize_percentile(y, pmin, pmax)

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
