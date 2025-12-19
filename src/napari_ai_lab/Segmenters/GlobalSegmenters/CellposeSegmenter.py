"""
Cellpose Global Segmenter.

This module provides a Cellpose segmenter for automatic cell segmentation
of entire images without user prompts.
"""

from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import cellpose at module level
try:
    import cellpose
    from cellpose import models

    _is_cellpose_available = True
    _cellpose_major_version = cellpose.version.split(".")[0]
except ImportError:
    cellpose = None
    models = None
    _is_cellpose_available = False
    _cellpose_major_version = None


@dataclass
class CellposeSegmenter(GlobalSegmenterBase):
    """
    Cellpose global segmenter.

    This segmenter uses the Cellpose model for automatic cell segmentation
    on entire images. Supports both Cellpose v3 and v4 APIs.
    """

    instructions = """
Cellpose Automatic Cell Segmentation:
• Automatically segments cells in the entire image
• Model Type: Choose based on your cell type (cyto, cyto2, nuclei, etc.)
• Diameter: Expected cell diameter in pixels (0 = auto-detect)
• GPU: Use GPU acceleration if available
• Flow Threshold: Controls segmentation sensitivity (higher = more conservative)
• Cellpose Iterations: Number of refinement iterations (more = slower but better)
• Best for: cells, nuclei, organisms with clear boundaries
• Works with both brightfield and fluorescence images
    """

    # Parameters for Cellpose segmentation
    model_type: str = field(
        default="cyto2",
        metadata={
            "type": "str",
            "choices": ["cyto", "cyto2", "nuclei", "cyto3"],
            "default": "cyto2",
        },
    )

    diameter: float = field(
        default=30.0,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 500.0,
            "step": 5.0,
            "default": 30.0,
        },
    )

    use_gpu: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "default": True,
        },
    )

    # Probability threshold (cell probability)
    prob_threshold: float = field(
        default=0.0,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.0,
        },
    )

    flow_threshold: float = field(
        default=0.4,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
            "default": 0.4,
        },
    )

    cellpose_iterations: int = field(
        default=200,
        metadata={
            "type": "int",
            "min": 50,
            "max": 5000,
            "step": 50,
            "default": 200,
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
            list: Supported axis configurations for Cellpose segmentation.
        """
        return ["YX", "YXC", "ZYX", "ZYXC", "TYX", "TYXC"]

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if cellpose can be imported, False otherwise.
        """
        return _is_cellpose_available

    def get_version(self):
        """Get the cellpose version."""
        if _is_cellpose_available:
            return "cellpose" + str(cellpose.version)
        return None

    def segment(self, image, **kwargs):
        """
        Perform Cellpose segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray or str: Labeled segmentation mask if dependencies available,
                                 otherwise execution string for remote processing.

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if len(image.shape) < 2:
            raise ValueError(
                f"CellposeSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Create model using cached major version
        try:
            if _cellpose_major_version == "4":
                model = models.CellposeModel(
                    gpu=self.use_gpu, model_type=self.model_type
                )
            else:
                model = models.Cellpose(
                    gpu=self.use_gpu, model_type=self.model_type
                )
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Error creating model with GPU, falling back to CPU: {e}")
            if _cellpose_major_version == "4":
                model = models.CellposeModel(
                    gpu=False, model_type=self.model_type
                )
            else:
                model = models.Cellpose(gpu=False, model_type=self.model_type)

        # Perform segmentation
        diameter = None if self.diameter == 0 else self.diameter

        if _cellpose_major_version == "3":
            masks, flows, styles, diams = model.eval(
                image,
                diameter=diameter,
                cellprob_threshold=self.prob_threshold,
                flow_threshold=self.flow_threshold,
                niter=self.cellpose_iterations,
            )
        else:
            masks, flows, styles = model.eval(
                image,
                diameter=diameter,
                cellprob_threshold=self.prob_threshold,
                flow_threshold=self.flow_threshold,
                niter=self.cellpose_iterations,
            )

        return masks.astype(np.uint16)

    def get_execution_string(self, image, **kwargs):
        """
        Generate a string containing the cellpose execution code for remote execution.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Python code string that can be executed in a cellpose environment.
        """
        # Get image properties
        image_shape = image.shape
        image_dtype = str(image.dtype)
        # Create the execution string
        execution_code = """
import numpy as np
task.outputs["hello"] = "Hello from CellposeSegmenter!"
"""

        # Create the execution string
        execution_code = f'''
import numpy as np
import cellpose
from cellpose import models

# Parameters from segmenter
model_type = "{self.model_type}"
use_gpu = {self.use_gpu}
diameter = {self.diameter if self.diameter != 0 else "None"}
flow_threshold = {self.flow_threshold}
prob_threshold = {self.prob_threshold}
cellpose_iterations = {self.cellpose_iterations}

# Image will be provided as 'image' variable
# image.shape = {image_shape}
# image.dtype = {image_dtype}

def cellpose_segment_remote(image):
    """Remote cellpose segmentation function."""

    # Get cellpose version and create appropriate model
    major_number = cellpose.version.split('.')[0]
    print(f"Cellpose version: {{cellpose.version}} (major number: {{major_number}})")

    try:
        if major_number == '4':
            model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        else:
            # For version 3 and all other versions, use the older API
            model = models.Cellpose(gpu=use_gpu, model_type=model_type)

        print(f"Created Cellpose model: {{model_type}}, GPU: {{use_gpu}}")

    except Exception as e:
        print(f"Error creating Cellpose model: {{e}}")
        print("Falling back to CPU mode...")
        try:
            if major_number == '4':
                model = models.CellposeModel(gpu=False, model_type=model_type)
            else:
                # For version 3 and all other versions, use the older API
                model = models.Cellpose(gpu=False, model_type=model_type)
        except Exception as fallback_error:
            raise RuntimeError(f"Could not create Cellpose model: {{fallback_error}}")

    # Perform segmentation
    diameter_value = None if diameter == "None" else diameter

    print(f"Cellpose: Segmenting image, diameter={{diameter_value}}, prob_threshold={{prob_threshold}}, flow_threshold={{flow_threshold}}")

    if major_number == '3':
        masks, flows, styles, diams = model.eval(
            image.ndarray(),
            diameter=diameter_value,
            cellprob_threshold=prob_threshold,
            flow_threshold=flow_threshold,
            niter=cellpose_iterations
            )
    else:
        masks, flows, styles = model.eval(
            image.ndarray(),
            diameter=diameter_value,
            cellprob_threshold=prob_threshold,
            flow_threshold=flow_threshold,
            niter=cellpose_iterations
    )

    print(f"Cellpose: Found {{len(np.unique(masks)) - 1}} cells")
    if diameter_value is None:
        print(f"Cellpose: Auto-detected diameter: {{diams}}")

    return masks.astype(np.uint16)

# Execute segmentation
result = cellpose_segment_remote(image)

# Convert result to appose format for output
import appose

ndarr_mask = appose.NDArray(dtype=str(result.dtype), shape=result.shape)
ndarr_mask.ndarray()[:] = result

task.outputs["mask"] = ndarr_mask
'''

        return execution_code

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {
            "model_type": self.model_type,
            "diameter": self.diameter,
            "use_gpu": self.use_gpu,
            "prob_threshold": self.prob_threshold,
            "flow_threshold": self.flow_threshold,
            "cellpose_iterations": self.cellpose_iterations,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("CellposeSegmenter", cls)
