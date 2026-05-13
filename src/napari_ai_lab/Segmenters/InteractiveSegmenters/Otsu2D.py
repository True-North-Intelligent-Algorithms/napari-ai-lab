"""
Otsu2D Interactive Segmenter.

This module provides a 2D Otsu thresholding segmenter that works with
2D grayscale and 2D color images.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage import filters, morphology

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
1. Click a point on the region of interest to threshold only a local ROI.
2. Lateral ROI Size controls the side length (pixels) of the square ROI centred on the point.
3. Without a point, Otsu thresholding is applied to the entire image.
4. The returned mask is always full-sized; non-ROI pixels are left as zero.
5. Supports both grayscale and color images (converts color to grayscale).
6. Best results with high contrast images (cells, particles, etc.).
    """

    lateral_roi_size: int = field(
        default=30,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 500,
            "step": 1,
            "default": 30,
        },
    )

    apply_opening: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    apply_closing: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    element_size: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 1,
            "max": 50,
            "step": 1,
            "default": 1,
        },
    )

    def __init__(self):
        """Initialize the Otsu2D segmenter."""
        super().__init__()
        self._supported_axes = ["YX", "YXC"]
        self._potential_axes = ["YX"]

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform Otsu thresholding segmentation on 2D image.

        When points are provided the last point's Y and X coordinates are used
        to define a square ROI of side ``lateral_roi_size``.  Otsu thresholding
        is applied only within that ROI and the resulting mask is placed back
        into a full-sized zero array.

        If no points are given the whole image is thresholded.

        Args:
            image (numpy.ndarray): Input 2D image to segment.
            points (list, optional): List of annotation points.  The last point
                is used; its [-2] and [-1] elements are the Y and X coordinate.
            shapes (list, optional): Ignored.
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
            if image.shape[2] == 3:  # RGB
                image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                image_gray = np.mean(image, axis=2)
        else:
            image_gray = image.copy()

        h, w = image_gray.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Determine ROI (priority: shape/box > point > full image)
        y0 = y1 = x0 = x1 = None

        if shapes is not None and len(shapes) > 0:
            # Use the last shape; treat it as a bounding box.
            # Last 2 columns of vertices are Y, X.
            box = np.asarray(shapes[-1])
            ys = box[:, -2]
            xs = box[:, -1]
            y0 = max(0, int(np.floor(np.min(ys))))
            y1 = min(h, int(np.ceil(np.max(ys))))
            x0 = max(0, int(np.floor(np.min(xs))))
            x1 = min(w, int(np.ceil(np.max(xs))))
            print(f"Otsu2D: using box ROI [{y0}:{y1}, {x0}:{x1}]")
        elif points is not None and len(points) > 0:
            # Use the last point; [-2] is Y, [-1] is X
            pt = np.asarray(points[-1])
            cy, cx = int(pt[-2]), int(pt[-1])
            half = self.lateral_roi_size // 2

            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)

        if y0 is not None:
            roi = image_gray[y0:y1, x0:x1]

            if roi.size == 0:
                print("Otsu2D: ROI is empty, returning empty mask")
                return full_mask

            threshold = filters.threshold_otsu(roi)
            roi_mask = (roi > threshold).astype(np.uint8)
            roi_mask = self._apply_post_processing(roi_mask)
            full_mask[y0:y1, x0:x1] = roi_mask

            print(
                f"Otsu2D: ROI [{y0}:{y1}, {x0}:{x1}] threshold={threshold:.2f}"
            )
        else:
            # No points or shapes — threshold the whole image
            threshold = filters.threshold_otsu(image_gray)
            full_mask = (image_gray > threshold).astype(np.uint8)
            full_mask = self._apply_post_processing(full_mask)
            print(f"Otsu2D: Full-image threshold {threshold:.2f}")

        return full_mask

    def _apply_post_processing(self, mask: np.ndarray) -> np.ndarray:
        """Apply binary open and/or close morphology to mask (open first)."""
        if not self.apply_opening and not self.apply_closing:
            return mask
        footprint = morphology.disk(self.element_size)
        result = mask.astype(bool)
        if self.apply_opening:
            result = morphology.binary_opening(result, footprint)
        if self.apply_closing:
            result = morphology.binary_closing(result, footprint)
        return result.astype(np.uint8)

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("Otsu2D", cls)
