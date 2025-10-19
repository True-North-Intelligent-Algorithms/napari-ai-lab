"""
SAM3D Interactive Segmenter.

This module provides a 3D Segment Anything Model (SAM) segmenter that works with
3D volumetric data using point and shape annotations.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator.util import prompt_segmentation

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
        default=0.7,
        metadata={
            "type": "float",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1.0,
            "default": 0.7,
            "step": 0.1,
        },
    )

    box_extension: float = field(
        default=0.5,
        metadata={
            "type": "float",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 2.0,
            "default": 0.5,
            "step": 0.1,
        },
    )

    def __init__(self):
        """Initialize the SAM3D segmenter."""
        super().__init__()

        self.state = None

    def initialize_predictor(self, image, save_path: str, image_name: str):
        """
        Initialize the SAM predictor with embeddings.

        Args:
            image: Current image data for embedding generation
            save_path (str): Directory path where embeddings are saved/loaded
            image_name (str): Name of the image (without extension)
        """
        print(
            f"SAM3D: initialize_predictor called with image_shape={image.shape}, save_path={save_path}, image_name={image_name}"
        )

        # Show message box asking about embeddings
        from qtpy.QtWidgets import QMessageBox

        # Create embedding directory path using Path
        parent_directory = Path(save_path).parent
        embedding_directory = parent_directory / "embeddings"
        embedding_save_path = embedding_directory / image_name

        # Create message with embedding save path information
        message = f"SAM3D segmenter has been called. Do you want to generate embedding at:\n{embedding_save_path}?"

        reply = QMessageBox.question(
            None,
            "SAM3D Segmenter",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            print("User chose to generate embeddings")
            self.embedding_directory = str(embedding_directory)
            self.embedding_save_path = str(embedding_save_path)
            print(f"SAM3D: Embedding save path: {self.embedding_save_path}")
            # TODO: Implement embedding generation/loading logic here
        else:
            print("User chose not to generate embeddings")
            self.embedding_directory = None
            self.embedding_save_path = None

        self.state = AnnotatorState()
        self.state.reset_state()
        self.state.initialize_predictor(
            image,
            model_type="vit_b_lm",
            ndim=3,
            save_path=self.embedding_save_path,
        )

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

        # Check if predictor has been initialized
        if (
            not hasattr(self, "embedding_directory")
            or self.embedding_directory is None
        ):
            print(
                "SAM3D: Predictor not initialized or user declined embeddings"
            )
            return None

        # TODO: Implement SAM3D segmentation logic
        # For now, return empty segmentation for GUI testing
        print(f"SAM3D: Processing image with shape {image.shape}")
        print(f"SAM3D: IoU Threshold = {self.iou_threshold}")
        print(f"SAM3D: Box Extension = {self.box_extension}")

        if points:
            print(f"SAM3D: Using {len(points)} points for segmentation")
        if shapes:
            print(f"SAM3D: Using {len(shapes)} shapes for segmentation")

        labels = [1] * len(points)

        z_pos = int(points[0][0])
        points_ = [
            (p[1], p[2]) for p in points
        ]  # Adjust points to 2D slice coordinates

        # perform prompt segmentation passing in the predictor, image_embeddings and points.
        seg_z_pos = prompt_segmentation(
            self.state.predictor,
            np.array(points_),
            np.array(labels),
            [],
            [],
            image.shape[1:],
            multiple_box_prompts=False,
            image_embeddings=self.state.image_embeddings,
            i=z_pos,
        )

        seg = np.zeros_like(image, dtype=np.uint8)
        seg[z_pos] = seg_z_pos

        return seg

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("SAM3D", cls)
