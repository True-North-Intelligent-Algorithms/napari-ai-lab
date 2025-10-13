"""
Otsu3D Interactive Segmenter.

This module provides a 3D Otsu thresholding segmenter that works with
3D volumetric data and 3D multichannel images.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage import filters

from .InteractiveSegmenterBase import InteractiveSegmenterBase


@dataclass
class Otsu3D(InteractiveSegmenterBase):
    """
    3D Otsu thresholding segmenter.

    This segmenter applies Otsu's automatic threshold selection method
    to create binary segmentation masks for 3D volumetric images.
    """

    instructions = """
Instructions for Otsu 3D Segmentation:
1. Automatic thresholding for 3D volumetric data (Z-stacks, time series)
2. Lateral ROI Size: Controls XY region analysis (20-50 pixels typical)
3. Axial ROI Size: Controls Z-direction analysis (5-15 slices typical)
4. Two modes available via advanced options:
   • Global: Single threshold for entire volume (default)
   • Slice-wise: Individual threshold per Z-slice (better for varying contrast)
5. Works with both grayscale volumes and multichannel (converts to grayscale)
6. Points and shapes are ignored - fully automatic segmentation
7. Ideal for: cell nuclei, fluorescent particles, consistent 3D structures
8. Adjust ROI sizes if results are too fragmented or merged
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

    axial_roi_size: int = field(
        metadata={
            "type": "int",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 1,
            "max": 100,
            "default": 10,
            "step": 1,
        }
    )

    def __init__(self, name=None):
        """
        Initialize the Otsu3D segmenter.

        Args:
            name (str, optional): Name of this segmenter instance.
        """
        super().__init__(name)

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: Supported axis configurations for 3D Otsu segmentation.
        """
        return ["ZYX", "ZYXC"]

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform Otsu thresholding segmentation on 3D image.

        Args:
            image (numpy.ndarray): Input 3D image to segment.
            points (list, optional): List of annotation points (ignored for Otsu).
            shapes (list, optional): List of annotation shapes (ignored for Otsu).
            **kwargs: Additional keyword arguments.
                - use_multichannel (bool): If True and image is multichannel,
                  convert to grayscale first. Default: True.
                - slice_wise (bool): If True, apply Otsu slice by slice instead
                  of globally. Default: False.

        Returns:
            numpy.ndarray: Binary segmentation mask (same shape as input image).

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) not in [3, 4]:
            raise ValueError(
                f"Otsu3D only supports 3D images. Got shape: {image.shape}"
            )

        # Handle multichannel images by converting to grayscale
        use_multichannel = kwargs.get("use_multichannel", True)
        slice_wise = kwargs.get("slice_wise", False)

        if len(image.shape) == 4 and use_multichannel:
            # Convert ZYXC to ZYX using weighted average
            if image.shape[3] == 3:  # RGB channels
                # Use standard RGB to grayscale conversion weights
                image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                # For other multichannel images, use simple average
                image_gray = np.mean(image, axis=3)
        else:
            image_gray = image.copy()

        if slice_wise:
            # Apply Otsu thresholding slice by slice (Z dimension)
            binary_mask = np.zeros_like(image_gray, dtype=bool)
            thresholds = []

            for z in range(image_gray.shape[0]):
                slice_2d = image_gray[z]
                threshold = filters.threshold_otsu(slice_2d)
                binary_mask[z] = slice_2d > threshold
                thresholds.append(threshold)

            print(
                f"Otsu3D: Applied slice-wise thresholding with {len(thresholds)} thresholds"
            )
            print(
                f"  Threshold range: {min(thresholds):.2f} - {max(thresholds):.2f}"
            )
            print(
                f"  ROI Settings - Lateral: {self.lateral_roi_size}, Axial: {self.axial_roi_size}"
            )
        else:
            # Apply global Otsu thresholding across entire 3D volume
            threshold = filters.threshold_otsu(image_gray)
            binary_mask = image_gray > threshold

            print(
                f"Otsu3D: Applied global threshold {threshold:.2f} to 3D volume"
            )
            print(
                f"  ROI Settings - Lateral: {self.lateral_roi_size}, Axial: {self.axial_roi_size}"
            )

        return binary_mask.astype(np.uint8)

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("Otsu3D", cls)
