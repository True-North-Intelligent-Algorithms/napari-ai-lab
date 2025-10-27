"""
StarDist Global Segmenter.

This module provides a StarDist segmenter for automatic segmentation
of entire 2D images without user prompts.
"""

from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import stardist at module level
try:
    import stardist
    from stardist.models import StarDist2D

    _is_stardist_available = True
except ImportError:
    stardist = None
    StarDist2D = None
    _is_stardist_available = False


@dataclass
class StardistSegmenter(GlobalSegmenterBase):
    """
    StarDist global segmenter (2D).

    Uses a pretrained StarDist2D model to segment entire images automatically.
    """

    instructions = """
StarDist Automatic Segmentation (2D):
• Automatically segments objects in the entire 2D image
• Model Preset: choose a pretrained model (e.g., 2D_versatile_fluo, 2D_versatile_he)
• Probability Threshold (prob_thresh): minimum object probability
• NMS Threshold (nms_thresh): non-maximum suppression IoU threshold
• Best for: nuclei, cells, star-convex objects in microscopy images
    """

    # Parameters for StarDist segmentation
    model_preset: str = field(
        default="2D_versatile_fluo",
        metadata={
            "type": "str",
            "choices": [
                "2D_versatile_fluo",
                "2D_versatile_he",
            ],
            "default": "2D_versatile_fluo",
        },
    )

    prob_thresh: float = field(
        default=0.5,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.5,
        },
    )

    nms_thresh: float = field(
        default=0.4,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.4,
        },
    )

    normalize_input: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "default": True,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

    @property
    def supported_axes(self):
        """
        Supported axis configurations for StarDist2D.

        Returns:
            list: Supported axis configurations.
        """
        # StarDist2D supports YX and YXC; allow T variants where the caller provides current slice
        return ["YX", "YXC", "TYX", "TYXC"]

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if stardist can be imported, False otherwise.
        """
        return _is_stardist_available

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Simple min-max normalization to [0, 1] with float32 output."""
        if image.dtype.kind in {"f"}:
            # Assume already in a reasonable range for floats
            return image.astype(np.float32, copy=False)
        img = image.astype(np.float32, copy=False)
        vmin = float(np.min(img))
        vmax = float(np.max(img))
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img

    def segment(self, image, **kwargs):
        """
        Perform StarDist segmentation on entire 2D image.

        Args:
            image (numpy.ndarray): Input image to segment (YX or YXC).
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray or str: Labeled segmentation mask if dependencies available,
                                 otherwise execution string for remote processing.

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if image.ndim < 2:
            raise ValueError(
                f"StardistSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        if not self.are_dependencies_available():
            return self.get_execution_string(image)

        # Ensure 2D input (YX or YXC). If higher dims were passed, caller should slice.
        if image.ndim > 3:
            raise ValueError(
                f"StardistSegmenter expects YX or YXC image. Got shape: {image.shape}"
            )

        # Optional normalization
        x = self._normalize(image) if self.normalize_input else image

        # Load pretrained StarDist model
        try:
            model = StarDist2D.from_pretrained(self.model_preset)
            print(f"Loaded StarDist2D pretrained model: {self.model_preset}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load StarDist2D model '{self.model_preset}': {e}"
            ) from e

        # Predict instances
        try:
            labels, details = model.predict_instances(
                x, prob_thresh=self.prob_thresh, nms_thresh=self.nms_thresh
            )
            print(
                f"StarDist: Found {len(np.unique(labels)) - 1} objects (prob_thresh={self.prob_thresh}, nms_thresh={self.nms_thresh})"
            )
        except Exception as e:
            raise RuntimeError(f"StarDist prediction failed: {e}") from e

        return labels.astype(np.uint16)

    def get_execution_string(self, image, **kwargs):
        """
        Generate a string containing the StarDist execution code for remote execution.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Python code string that can be executed in a StarDist-enabled environment.
        """
        image_shape = image.shape
        image_dtype = str(image.dtype)

        execution_code = f"""
import numpy as np
from stardist.models import StarDist2D

# Parameters from segmenter
model_preset = "{self.model_preset}"
prob_thresh = {self.prob_thresh}
nms_thresh = {self.nms_thresh}

# Image will be provided as 'image' variable
# image.shape = {image_shape}
# image.dtype = {image_dtype}

def _normalize(img):
    img = img.astype(np.float32, copy=False)
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img

def stardist_segment_remote(image):
    x = _normalize(image)
    model = StarDist2D.from_pretrained(model_preset)
    labels, details = model.predict_instances(x, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    return labels.astype(np.uint16)

# Execute segmentation
result = stardist_segment_remote(image.ndarray())

# Convert result to appose format for output
import appose

ndarr_mask = appose.NDArray(dtype=str(result.dtype), shape=result.shape)
ndarr_mask.ndarray()[:] = result

task.outputs["mask"] = ndarr_mask
"""

        print(
            "StarDist not available locally - generated execution string for remote processing"
        )
        return execution_code

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {
            "model_preset": self.model_preset,
            "prob_thresh": self.prob_thresh,
            "nms_thresh": self.nms_thresh,
            "normalize_input": self.normalize_input,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("StardistSegmenter", cls)
