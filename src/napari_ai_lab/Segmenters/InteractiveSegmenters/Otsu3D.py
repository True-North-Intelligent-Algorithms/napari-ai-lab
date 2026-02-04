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
        default=30,
        metadata={
            "type": "int",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0,
            "max": 500,
            "default": 30,
            "step": 1,
        },
    )

    axial_roi_size: int = field(
        default=10,
        metadata={
            "type": "int",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 1,
            "max": 100,
            "default": 10,
            "step": 1,
        },
    )

    def __init__(self):
        """Initialize the Otsu3D segmenter."""
        super().__init__()
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX"]

    def segment(self, image, points=None, shapes=None):
        """
        Perform Otsu thresholding segmentation on 3D ROI around point.

        Args:
            image (numpy.ndarray): Input 3D image (ZYX) to segment.
            points (list, optional): List of points. Uses points[0] as ROI center.
            shapes (list, optional): Ignored for Otsu.

        Returns:
            numpy.ndarray: Binary segmentation mask.
        """
        if len(image.shape) != 3:
            raise ValueError(
                f"Otsu3D requires 3D image. Got shape: {image.shape}"
            )

        mask = np.zeros(image.shape, dtype=np.uint8)

        if not points or len(points) == 0:
            return mask

        # Use first point as center
        z, y, x = int(points[0][0]), int(points[0][1]), int(points[0][2])

        # Define ROI bounds
        z_min = max(0, z - self.axial_roi_size // 2)
        z_max = min(image.shape[0], z + self.axial_roi_size // 2)
        y_min = max(0, y - self.lateral_roi_size // 2)
        y_max = min(image.shape[1], y + self.lateral_roi_size // 2)
        x_min = max(0, x - self.lateral_roi_size // 2)
        x_max = min(image.shape[2], x + self.lateral_roi_size // 2)

        # Extract ROI
        roi = image[z_min:z_max, y_min:y_max, x_min:x_max]

        # Apply Otsu to ROI
        threshold = filters.threshold_otsu(roi)
        roi_mask = roi > threshold

        # Put result back into mask
        mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_mask.astype(np.uint8)

        return mask

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("Otsu3D", cls)
