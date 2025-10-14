"""
Otsu Global Segmenter.

This module provides an Otsu thresholding segmenter for automatic
segmentation of entire images without user prompts.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage import filters

from .GlobalSegmenterBase import GlobalSegmenterBase


@dataclass
class OtsuSegmenter(GlobalSegmenterBase):
    """
    Otsu thresholding global segmenter.

    This segmenter applies Otsu's automatic threshold selection method
    to create binary segmentation masks for entire images automatically.
    No user interaction required.
    """

    instructions = """
Otsu Automatic Segmentation:
• Automatically finds optimal threshold for the entire image
• Works best with bimodal intensity distributions (clear foreground/background)
• Converts color images to grayscale automatically
• Produces binary masks (0 for background, 1 for foreground)
• Best for: cells, particles, text, high-contrast objects
• May struggle with: gradual transitions, low contrast, noisy images
    """

    # Parameters for Otsu segmentation
    invert_mask: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "default": False,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: Supported axis configurations for Otsu segmentation.
        """
        return ["YX", "YXC", "ZYX", "ZYXC", "TYX", "TYXC", "TZYX", "TZYXC"]

    def segment(self, image, **kwargs):
        """
        Perform Otsu thresholding segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Binary segmentation mask (same shape as input image).

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) < 2:
            raise ValueError(
                f"OtsuSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Handle different image dimensions
        if len(image.shape) == 2:
            # Simple 2D grayscale image
            processed_image = self._segment_2d(image)
        elif len(image.shape) == 3:
            # Could be YXC (2D with channels) or ZYX (3D grayscale)
            if image.shape[-1] <= 4:  # Assume last dim is channels if small
                # 2D multichannel image (YXC)
                processed_image = self._segment_2d_multichannel(image)
            else:
                # 3D grayscale image (ZYX) - process slice by slice
                processed_image = self._segment_3d(image)
        elif len(image.shape) == 4:
            # Could be ZYXC or TYX, etc. - process slice by slice
            processed_image = self._segment_4d(image)
        else:
            raise ValueError(
                f"OtsuSegmenter does not support {len(image.shape)}D images"
            )

        return processed_image

    def _segment_2d(self, image):
        """Segment a 2D grayscale image."""
        threshold = filters.threshold_otsu(image)
        binary_mask = image > threshold

        if self.invert_mask:
            binary_mask = ~binary_mask

        print(f"Otsu: Applied threshold {threshold:.2f}")
        return binary_mask.astype(np.uint8)

    def _segment_2d_multichannel(self, image):
        """Segment a 2D multichannel image by converting to grayscale first."""
        # Convert to grayscale
        if image.shape[2] == 3:  # RGB
            grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            grayscale = np.mean(image, axis=2)

        return self._segment_2d(grayscale)

    def _segment_3d(self, image):
        """Segment a 3D image slice by slice."""
        result = np.zeros_like(image, dtype=np.uint8)

        for z in range(image.shape[0]):
            slice_2d = image[z, :, :]
            result[z, :, :] = self._segment_2d(slice_2d)

        print(f"Otsu: Processed {image.shape[0]} slices")
        return result

    def _segment_4d(self, image):
        """Segment a 4D image (e.g., TZYX or ZYXC) slice by slice."""
        result = np.zeros_like(image, dtype=np.uint8)

        if image.shape[-1] <= 4:  # Assume ZYXC
            for z in range(image.shape[0]):
                slice_3d = image[z, :, :, :]
                result[z, :, :, :] = self._segment_2d_multichannel(slice_3d)
        else:  # Assume TZYX or similar
            for t in range(image.shape[0]):
                slice_3d = image[t, :, :, :]
                result[t, :, :, :] = self._segment_3d(slice_3d)

        print(f"Otsu: Processed 4D image with shape {image.shape}")
        return result

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {"invert_mask": self.invert_mask}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("OtsuSegmenter", cls)
