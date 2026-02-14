"""Deep learning utility functions."""

import numpy as np


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
