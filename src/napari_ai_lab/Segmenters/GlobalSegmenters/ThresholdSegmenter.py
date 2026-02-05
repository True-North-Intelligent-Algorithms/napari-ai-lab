"""
Threshold Global Segmenter.

This module provides a simple threshold segmenter for automatic
segmentation with user-defined threshold values.
"""

from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase


@dataclass
class ThresholdSegmenter(GlobalSegmenterBase):
    """
    Simple threshold global segmenter.

    This segmenter applies a user-defined threshold to create binary
    segmentation masks for entire images automatically.
    """

    instructions = """
Simple Threshold Segmentation:
• Apply a fixed threshold value to the entire image
• Threshold: Intensity value to separate foreground/background
• Invert Mask: Swap foreground and background regions
• Works with any image type (converts multichannel to grayscale)
• Best for: Images where you know the optimal threshold value
• Faster than Otsu when threshold is known in advance
    """

    # Parameters for threshold segmentation
    threshold: float = field(
        default=128.0,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 65535.0,
            "step": 1.0,
            "default": 128.0,
        },
    )

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

        # Set supported axes
        self._supported_axes = ["YX", "YXC"]
        self._potential_axes = ["YX", "YXC"]

    def segment(self, image, **kwargs):
        """
        Perform threshold segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Binary segmentation mask.
        """
        if len(image.shape) < 2:
            raise ValueError(
                f"ThresholdSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Convert to working image
        working_image = self._prepare_image(image)

        # Apply threshold
        binary_mask = working_image > self.threshold

        if self.invert_mask:
            binary_mask = ~binary_mask

        print(f"Threshold: Applied threshold {self.threshold}")
        return binary_mask.astype(np.uint8)

    def _prepare_image(self, image):
        """Prepare image for thresholding by handling multichannel conversion."""
        # Handle multichannel images by converting to grayscale
        if len(image.shape) >= 3 and image.shape[-1] <= 4:
            # Assume last dimension is channels if it's small
            if image.shape[-1] == 3:  # RGB
                # Use standard RGB to grayscale conversion
                channels = image.reshape(-1, 3)
                grayscale_flat = np.dot(channels, [0.2989, 0.5870, 0.1140])
                grayscale = grayscale_flat.reshape(image.shape[:-1])
            else:
                # For other channel counts, use simple average
                grayscale = np.mean(image, axis=-1)
            return grayscale
        else:
            # Already grayscale or treat as grayscale
            return image.copy()

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {"threshold": self.threshold, "invert_mask": self.invert_mask}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework(
            "ThresholdSegmenter", cls
        )
