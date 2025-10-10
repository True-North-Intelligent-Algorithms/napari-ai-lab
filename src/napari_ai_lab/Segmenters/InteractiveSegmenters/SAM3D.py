"""
SAM3D Interactive Segmenter.

This module provides a 3D Segment Anything Model (SAM) segmenter that works with
3D volumetric data using point and shape annotations.
"""

from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase


@dataclass
class SAM3D(InteractiveSegmenterBase):
    """
    3D Segment Anything Model (SAM) segmenter.

    This segmenter uses point and shape annotations to create precise
    3D segmentation masks using the SAM architecture.
    """

    instructions = """
Instructions:
1. Activate Points layer
2. Draw points on objects
3. SAM creates 3D segmentation
4. Press 'C' to commit current label
5. Press 'X' to erase current label
6. Press 'V' to toggle positive/negative points
    """

    iou_threshold: float = field(
        metadata={
            "type": "float",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1.0,
            "default": 0.7,
            "step": 0.1,
        }
    )

    box_extension: float = field(
        metadata={
            "type": "float",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 2.0,
            "default": 0.5,
            "step": 0.1,
        }
    )

    def __init__(self, name=None):
        """
        Initialize the SAM3D segmenter.

        Args:
            name (str, optional): Name of this segmenter instance.
        """
        super().__init__(name)

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: Supported axis configurations for 3D SAM segmentation.
        """
        return ["ZYX", "ZYXC"]

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform SAM segmentation on 3D image using points and shapes.

        Args:
            image (numpy.ndarray): Input 3D image to segment.
            points (list, optional): List of annotation points for prompting SAM.
            shapes (list, optional): List of annotation shapes for prompting SAM.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Segmentation mask (same shape as input image).

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) not in [3, 4]:
            raise ValueError(
                f"SAM3D only supports 3D images. Got shape: {image.shape}"
            )

        # TODO: Implement SAM3D segmentation logic
        # For now, return empty segmentation for GUI testing
        print(f"SAM3D: Processing image with shape {image.shape}")
        print(f"SAM3D: IoU Threshold = {self.iou_threshold}")
        print(f"SAM3D: Box Extension = {self.box_extension}")

        if points:
            print(f"SAM3D: Using {len(points)} points for segmentation")
        if shapes:
            print(f"SAM3D: Using {len(shapes)} shapes for segmentation")

        # Return empty segmentation mask for now
        if len(image.shape) == 4:
            return np.zeros(image.shape[:3], dtype=np.uint8)
        else:
            return np.zeros(image.shape, dtype=np.uint8)


# Register this segmenter when the module is imported
InteractiveSegmenterBase.register_framework("SAM3D", SAM3D)
