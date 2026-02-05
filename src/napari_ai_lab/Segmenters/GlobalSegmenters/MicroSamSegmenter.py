"""
MicroSam Global Segmenter.

This module provides a MicroSam segmenter for automatic instance segmentation
of entire images without user prompts.
"""

from dataclasses import dataclass, field

import numpy as np

from ...utility import get_ndim
from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import micro_sam at module level
try:
    import micro_sam
    import torch
    from micro_sam.automatic_segmentation import (
        automatic_instance_segmentation,
        get_predictor_and_segmenter,
    )

    _is_microsam_available = True
except ImportError:
    micro_sam = None
    _is_microsam_available = False


@dataclass
class MicroSamSegmenter(GlobalSegmenterBase):
    """
    MicroSam global segmenter.

    This segmenter uses the MicroSam model for automatic instance segmentation
    on entire images.
    """

    instructions = """
MicroSam Automatic Instance Segmentation:
• Automatically segments instances in the entire image
• Model Type: Choose SAM model variant (vit_b, vit_l, vit_h, vit_b_lm, etc.)
• Tile Shape: Set to None for non-tiled, or (384, 384) for tiled processing
• Halo: Overlap between tiles (default: 64, 64)
• GPU: Automatically uses CUDA if available
• Best for: microscopy images, cells, nuclei
    """

    # Parameters for MicroSam segmentation
    model_type: str = field(
        default="vit_b_lm",
        metadata={
            "type": "str",
            "choices": ["vit_b", "vit_l", "vit_h", "vit_b_lm", "vit_t"],
            "default": "vit_b_lm",
        },
    )

    use_tiling: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "default": False,
        },
    )

    tile_width: int = field(
        default=384,
        metadata={
            "type": "int",
            "min": 128,
            "max": 1024,
            "step": 64,
            "default": 384,
        },
    )

    tile_height: int = field(
        default=384,
        metadata={
            "type": "int",
            "min": 128,
            "max": 1024,
            "step": 64,
            "default": 384,
        },
    )

    halo_x: int = field(
        default=64,
        metadata={
            "type": "int",
            "min": 0,
            "max": 256,
            "step": 16,
            "default": 64,
        },
    )

    halo_y: int = field(
        default=64,
        metadata={
            "type": "int",
            "min": 0,
            "max": 256,
            "step": 16,
            "default": 64,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

        # Set supported axes
        self._supported_axes = ["YX", "ZYX"]
        self._potential_axes = ["YX", "ZYX"]

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if micro_sam can be imported, False otherwise.
        """
        return _is_microsam_available

    def get_version(self):
        """Get the micro_sam version."""
        if _is_microsam_available:
            if hasattr(micro_sam, "__version__"):
                return "micro_sam" + str(micro_sam.__version__)
            return "micro_sam (version unknown)"
        return None

    def segment(self, image, **kwargs):
        """
        Perform MicroSam segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Labeled segmentation mask.

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) < 2:
            raise ValueError(
                f"MicroSamSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"MicroSam: Using device: {device}")

        # Determine tile shape
        tile_shape = None
        if self.use_tiling:
            tile_shape = (self.tile_height, self.tile_width)
            print(
                f"MicroSam: Using tiled segmentation with tile_shape={tile_shape}"
            )
        else:
            print("MicroSam: Using non-tiled segmentation")

        halo = (self.halo_y, self.halo_x)

        # Get predictor and segmenter
        print(f"MicroSam: Loading model type: {self.model_type}")
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=self.model_type,
            device=device,
            is_tiled=(tile_shape is not None),
        )

        ndim = get_ndim(image.shape)

        # Perform automatic instance segmentation
        print(f"MicroSam: Running segmentation (ndim={ndim})...")
        result = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
        )

        print(f"MicroSam: Found {len(np.unique(result)) - 1} instances")
        return result.astype(np.uint16)

    def _generate_execution_string(self, image):
        """Generate execution string for remote processing."""
        tile_shape_str = (
            f"({self.tile_height}, {self.tile_width})"
            if self.use_tiling
            else "None"
        )

        execution_code = f'''
# MicroSam Automatic Instance Segmentation
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import torch
import numpy as np

def microsam_segment_remote(image):
    """Perform MicroSam segmentation on image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tile_shape = {tile_shape_str}
    halo = ({self.halo_y}, {self.halo_x})
    model_type = "{self.model_type}"

    # Get predictor and segmenter
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        device=device,
        is_tiled=(tile_shape is not None),
    )

    # Determine ndim
    ndim = 2 if len(image.shape) == 2 else len(image.shape)

    # Perform segmentation
    result = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=ndim,
        tile_shape=tile_shape,
        halo=halo,
    )

    print(f"MicroSam: Found {{len(np.unique(result)) - 1}} instances")
    return result.astype(np.uint16)

# Execute segmentation
result = microsam_segment_remote(image)

# Convert result to appose format for output
import appose

ndarr_mask = appose.NDArray(dtype=str(result.dtype), shape=result.shape)
ndarr_mask.ndarray()[:] = result

task.outputs["mask"] = ndarr_mask
'''

        print(
            "MicroSam not available locally - generated execution string for remote processing"
        )
        return execution_code

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {
            "model_type": self.model_type,
            "use_tiling": self.use_tiling,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "halo_x": self.halo_x,
            "halo_y": self.halo_y,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("MicroSamSegmenter", cls)
