"""
SAM3D Interactive Segmenter.

This module provides a 3D Segment Anything Model (SAM) segmenter that works with
3D volumetric data using point and shape annotations.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from napari_ai_lab.utility import get_ndim

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator.util import prompt_segmentation

    _is_micro_sam_available = True
except ImportError:
    AnnotatorState = None
    _is_micro_sam_available = False
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
            "param_type": "inference",
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
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 2.0,
            "default": 0.5,
            "step": 0.1,
        },
    )

    model_type: str = field(
        default="vit_b",
        metadata={
            "type": "str",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "choices": [
                "vit_b",
                "vit_b_lm",
                "vit_b_em_organelles",
                "vit_b_medical_imaging",
                "vit_b_histopathology",
            ],
            "default": "vit_b",
        },
    )

    def __init__(self):
        """Initialize the SAM3D segmenter."""
        super().__init__()

        self.state = None

        self._supported_axes = ["ZYX", "YX", "ZYXC", "YXC"]
        self._potential_axes = ["ZYX", "YX", "ZYXC", "YXC"]
        self.selected_axis = self._supported_axes[0]

    def initialize_embedding_save_path(self, save_path: str, image_name: str):
        """
        Initialize the embedding save path for this segmenter.

        Args:
            save_path (str): Directory path where embeddings are saved/loaded
            image_name (str): Name of the image (without extension)
        """
        # Create embedding directory path using Path
        embedding_parent_path = Path(save_path).parent / "embeddings"
        embedding_save_path = (
            embedding_parent_path / image_name / self.model_type
        )

        self.embedding_parent_path = str(embedding_parent_path)
        self.embedding_save_path = str(embedding_save_path)

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

        # Initialize embedding save path
        self.initialize_embedding_save_path(save_path, image_name)

        self.state = AnnotatorState()
        self.state.reset_state()

        ndim = get_ndim(image.shape)

        try:
            self.state.initialize_predictor(
                image,
                model_type=self.model_type,
                ndim=ndim,
                save_path=self.embedding_save_path,
            )
        except Exception as e:  # noqa
            print(f"Error initializing predictor: {e}")
            import traceback

            traceback.print_exc()

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: Supported axis configurations for 3D SAM segmentation.
        """
        return self._supported_axes

    @supported_axes.setter
    def supported_axes(self, value):
        """
        Set the list of axis configurations this segmenter supports.

        Args:
            value (list): List of supported axis strings.
        """
        self._supported_axes = value

    @property
    def potential_axes(self):
        """
        Get the list of all axis configurations this algorithm could potentially support.

        Returns:
            list: Potential axis configurations for 3D SAM segmentation.
        """
        return self._potential_axes

    @potential_axes.setter
    def potential_axes(self, value):
        """
        Set the list of all axis configurations this algorithm could potentially support.

        Args:
            value (list): List of potential axis strings.
        """
        self._potential_axes = value

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if micro_sam can be imported, False otherwise.
        """
        return _is_micro_sam_available

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
        if len(image.shape) not in [2, 3, 4]:
            raise ValueError(
                f"SAM3D does not support 5D+ images. Got shape: {image.shape}"
            )

        # Check if predictor has been initialized
        if (
            self.embedding_parent_path is None
            or self.state is None
            or self.state.image_embeddings is None
        ):
            print("SAM3D: Predictor not initialized")
            return None

        print(f"SAM3D: Processing image with shape {image.shape}")

        if points is not None:
            print(f"SAM3D: Using {len(points)} points for segmentation")
        if shapes is not None:
            print(f"SAM3D: Using {len(shapes)} shapes for segmentation")

        # TODO assign labels based on point types (positive/negative)
        labels = [1] * len(points)

        # Handle both 2D and 3D points
        if len(points[0]) == 3:
            # 3D points (Z, Y, X)
            z_pos = int(points[0][0])
            points_ = [
                (p[1], p[2]) for p in points
            ]  # Extract Y, X from 3D points
        else:
            z_pos = None
            points_ = [
                (p[0], p[1]) for p in points
            ]  # Use Y, X directly from 2D points

        # perform prompt segmentation passing in the predictor, image_embeddings and points.
        seg_z_pos = prompt_segmentation(
            self.state.predictor,
            np.array(points_),
            np.array(labels),
            [],
            [],
            image.shape[-2:],
            multiple_box_prompts=False,
            image_embeddings=self.state.image_embeddings,
            i=z_pos,
        )

        if z_pos is not None:
            seg = np.zeros_like(image, dtype=np.uint8)
            seg[z_pos] = seg_z_pos
        else:
            seg = seg_z_pos

        print(self.selected_axis)

        # if selected axis is ZYX, perform full volume segmentation
        if self.selected_axis == "ZYX":
            stop_lower, stop_upper = False, False
            projection = "single_point"

            # Step 2: Segment the rest of the volume based on projecting prompts.
            seg, (z_min, z_max) = segment_mask_in_volume(
                seg,
                self.state.predictor,
                self.state.image_embeddings,
                np.array([z_pos]),
                stop_lower=stop_lower,
                stop_upper=stop_upper,
                iou_threshold=self.iou_threshold,
                projection=projection,
                box_extension=self.box_extension,
                # update_progress=lambda update: pbar_signals.pbar_update.emit(update),
            )

        return seg

    def get_execution_string(self, image, **kwargs):
        """
        Generate a string containing the SAM3D execution code for remote execution.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Python code string that can be executed in a micro_sam environment.
        """
        # Create the execution string with just imports for now
        execution_code = f"""

import numpy as np
from micro_sam.multi_dimensional_segmentation import segment_mask_in_volume
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.util import prompt_segmentation

embedding_save_path = r"{self.embedding_save_path}"

state = AnnotatorState()
state.reset_state()
state.initialize_predictor(
    image.ndarray(),
    model_type="{self.model_type}",
    ndim=len(image.ndarray().shape),
    save_path=embedding_save_path
)

# Handle both 2D and 3D points
if len(test_points[0]) == 3:
    # 3D points (Z, Y, X)
    z_pos = int(test_points[0][0])
    points_ = [
        (p[1], p[2]) for p in test_points
    ]  # Extract Y, X from 3D points
else:
    z_pos = None
    points_ = [
        (p[0], p[1]) for p in test_points
    ]  # Use Y, X directly from 2D points


labels = [1] * len(test_points)

# perform prompt segmentation passing in the predictor, image_embeddings and points.
seg_z_pos = prompt_segmentation(
    state.predictor,
    np.array(points_),
    np.array(labels),
    [],
    [],
    image.ndarray().shape[-2:],
    multiple_box_prompts=False,
    image_embeddings=state.image_embeddings,
    i=z_pos,
)

# Convert result to appose format for output
import appose
ndarr_mask = appose.NDArray(dtype='uint16', shape=image.ndarray().shape)

if z_pos is not None:
    ndarr_mask.ndarray()[z_pos,:] = seg_z_pos
else:
    ndarr_mask.ndarray()[:] = seg_z_pos

task.outputs['mask'] = ndarr_mask

"""

        print(
            "SAM3D not available locally - generated execution string for remote processing"
        )
        return execution_code

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("SAM3D", cls)
