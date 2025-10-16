"""
SAM3D Interactive Segmenter.

This module provides a 3D Segment Anything Model (SAM) segmenter that works with
3D volumetric data using point and shape annotations.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from micro_sam.sam_annotator._state import AnnotatorState

    is_micro_sam_available = True
except ImportError:
    AnnotatorState = None
    is_micro_sam_available = False
    print(
        "Warning: micro_sam is not installed. SAM3D segmenter will not work."
    )


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

    def __init__(self):
        """Initialize the SAM3D segmenter."""
        super().__init__()

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
            **kwargs: Additional keyword arguments including:
                parent_directory (str, optional): Directory path for embedding storage.

        Returns:
            numpy.ndarray: Segmentation mask (same shape as input image).

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) not in [3, 4]:
            raise ValueError(
                f"SAM3D only supports 3D images. Got shape: {image.shape}"
            )

        # Show message box asking about embeddings
        from qtpy.QtWidgets import QMessageBox

        # Extract parent directory from kwargs if provided
        parent_directory = kwargs.get("parent_directory", "unknown folder")

        # Create message with folder information
        message = f"SAM3D segmenter has been called. Do you want to generate embedding in folder {parent_directory}?"

        reply = QMessageBox.question(
            None,
            "SAM3D Segmenter",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            print("User chose to generate embeddings")
        else:
            print("User chose not to generate embeddings")

        self.embedding_directory = os.path.join(parent_directory, "embeddings")

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

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("SAM3D", cls)
