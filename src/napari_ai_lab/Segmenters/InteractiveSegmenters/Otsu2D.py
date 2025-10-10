"""
Otsu2D Interactive Segmenter.

This module provides a 2D Otsu thresholding segmenter that works with
2D grayscale and 2D color images.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage import filters

from .InteractiveSegmenterBase import InteractiveSegmenterBase


@dataclass
class Otsu2D(InteractiveSegmenterBase):
    """
    2D Otsu thresholding segmenter.

    This segmenter applies Otsu's automatic threshold selection method
    to create binary segmentation masks for 2D images.
    """

    instructions = """
Instructions for Otsu 2D Segmentation:
1. Works best with images that have clear foreground/background separation
2. Automatically finds optimal threshold - no manual tuning needed
3. Lateral ROI Size: Affects region analysis (typically 20-50 pixels)
4. Supports both grayscale and color images (converts color to grayscale)
5. Points and shapes are ignored - this is fully automatic thresholding
6. Best results with high contrast images (cells, particles, etc.)
7. May struggle with gradual intensity transitions or noisy images
    """

    lateral_roi_size: int = field(
        metadata={
            "type": "int",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0,
            "max": 500,
            "default": 30,
            "step": 1,
        }
    )

    def __init__(self, name=None):
        """
        Initialize the Otsu2D segmenter.

        Args:
            name (str, optional): Name of this segmenter instance.
        """
        super().__init__(name)

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: Supported axis configurations for 2D Otsu segmentation.
        """
        return ["YX", "YXC"]

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform Otsu thresholding segmentation on 2D image.

        Args:
            image (numpy.ndarray): Input 2D image to segment.
            points (list, optional): List of annotation points (ignored for Otsu).
            shapes (list, optional): List of annotation shapes (ignored for Otsu).
            **kwargs: Additional keyword arguments.
                - use_multichannel (bool): If True and image is multichannel,
                  convert to grayscale first. Default: True.

        Returns:
            numpy.ndarray: Binary segmentation mask (same shape as input image).

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) not in [2, 3]:
            raise ValueError(
                f"Otsu2D only supports 2D images. Got shape: {image.shape}"
            )

        # Handle multichannel images by converting to grayscale
        use_multichannel = kwargs.get("use_multichannel", True)
        if len(image.shape) == 3 and use_multichannel:
            # Convert RGB/multichannel to grayscale using weighted average
            if image.shape[2] == 3:  # RGB
                # Use standard RGB to grayscale conversion weights
                image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                # For other multichannel images, use simple average
                image_gray = np.mean(image, axis=2)
        else:
            image_gray = image.copy()

        # Apply Otsu thresholding
        threshold = filters.threshold_otsu(image_gray)
        binary_mask = image_gray > threshold

        print(
            f"Otsu2D: Applied threshold {threshold:.2f} to create binary mask"
        )

        return binary_mask.astype(np.uint8)


# Register this segmenter when the module is imported
InteractiveSegmenterBase.register_framework("Otsu2D", Otsu2D)
