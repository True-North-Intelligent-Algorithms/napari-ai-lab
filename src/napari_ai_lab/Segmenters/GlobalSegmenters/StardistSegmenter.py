"""
StarDist Global Segmenter.

This module provides a StarDist segmenter for automatic segmentation
of entire 2D images without user prompts.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import stardist at module level
try:
    import stardist
    from stardist.models import StarDist2D, StarDist3D

    _is_stardist_available = True
except ImportError:
    stardist = None
    StarDist2D = None
    StarDist3D = None
    _is_stardist_available = False


@dataclass
class StardistSegmenter(GlobalSegmenterBase):
    """
    StarDist global segmenter

    Uses a pretrained StarDist model to segment entire images automatically.
    """

    instructions = """
StarDist Automatic Segmentation:
• Automatically segments objects in the entire image
• Model Preset: choose a pretrained model (e.g., 2D_versatile_fluo, 2D_versatile_he)
• Alternatively, provide a custom model path
• Probability Threshold (prob_thresh): minimum object probability
• NMS Threshold (nms_thresh): non-maximum suppression IoU threshold
    """

    # Parameters for StarDist segmentation
    model_preset: str = field(
        default="2D_versatile_fluo",
        metadata={
            "type": "str",
            "param_type": "inference",
            "choices": [
                "2D_versatile_fluo",
                "2D_versatile_he",
                "3D_demo",
            ],
            "default": "2D_versatile_fluo",
        },
    )

    prob_thresh: float = field(
        default=0.5,
        metadata={
            "type": "float",
            "param_type": "inference",
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
            "param_type": "inference",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.4,
        },
    )

    model_path: str = field(
        default="",
        metadata={
            "type": "file",
            "param_type": "inference",
            "file_type": "directory",
            "default": "",
        },
    )

    normalize_input: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "default": True,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()
        # Initialize custom model storage

        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]

        self.custom_model = None
        self.is_3d_model = False

    def __setattr__(self, name, value):
        """Override setattr to detect model_path changes."""
        # Get old value if it exists
        old_value = getattr(self, name, None) if hasattr(self, name) else None

        # Set the new value
        super().__setattr__(name, value)

        # Check if this is model_path and value changed
        if name == "model_path" and old_value != value:
            print(
                f"🔄 Model path changed from '{old_value}' to '{value}'"
            )  # Debug print
            self._on_model_path_changed(value)

    def _on_model_path_changed(self, stardist_path: str):
        """Handle model path changes - load custom model and check if 2D or 3D."""
        print(f"📁 Loading custom StarDist model from: {stardist_path}")

        self.model_path = stardist_path
        model_base_path = os.path.dirname(stardist_path)
        model_name = os.path.basename(stardist_path)

        # Try to load as StarDist2D first
        try:
            model_2d = StarDist2D(
                None, name=model_name, basedir=model_base_path
            )
            print(
                f"✅ Successfully loaded 2D StarDist model from: {stardist_path}"
            )
            print(f"   Model config: {model_2d.config}")
            self.custom_model = model_2d
            self.is_3d_model = False
            return
        except (ValueError, FileNotFoundError, RuntimeError, TypeError) as e2d:
            print(f"   ❌ Failed to load as 2D model: {e2d}")

        # If 2D failed, try 3D
        try:
            model_3d = StarDist3D(
                None, name=model_name, basedir=model_base_path
            )
            print(
                f"✅ Successfully loaded 3D StarDist model from: {stardist_path}"
            )
            print(f"   Model config: {model_3d.config}")
            self.custom_model = model_3d
            self.is_3d_model = True
            print(
                "   ⚠️  Warning: This segmenter is designed for 2D - 3D model may not work properly"
            )
            return
        except (ValueError, FileNotFoundError, RuntimeError, TypeError) as e3d:
            print(f"   ❌ Failed to load as 3D model: {e3d}")

        # Both failed
        print(
            f"   ❌ Could not load model from {stardist_path} as either 2D or 3D StarDist model"
        )
        self.custom_model = None
        self.is_3d_model = False

        # Set supported axes
        self._supported_axes = ["YX", "ZYX"]
        self._potential_axes = ["YX", "ZYX"]

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
        Perform StarDist segmentation on 2D or 3D image.

        Args:
            image (numpy.ndarray): Input image to segment (YX, YXC for 2D or ZYX, ZYXC for 3D).
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Labeled segmentation mask.

        Raises:
            ValueError: If image dimensions are not supported.
        """
        if image.ndim < 2:
            raise ValueError(
                f"StardistSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Check if 3D model is selected
        is_3d_model = self.model_preset.startswith("3D_")

        # Use custom model if loaded, otherwise use preset
        model = None
        if hasattr(self, "custom_model") and self.custom_model is not None:
            print(f"Using custom StarDist model from: {self.model_path}")
            model = self.custom_model
        else:
            print(f"Using preset model: {self.model_preset}")

        # Validate dimensions based on model type
        if is_3d_model:
            if image.ndim < 3:
                raise ValueError(
                    f"3D StarDist model requires at least 3D images (ZYX). Got shape: {image.shape}"
                )
            # For 3D: expect ZYX or ZYXC
            if image.ndim > 4:
                raise ValueError(
                    f"StardistSegmenter expects ZYX or ZYXC image for 3D. Got shape: {image.shape}"
                )
        else:
            # For 2D: expect YX or YXC
            if image.ndim > 3:
                raise ValueError(
                    f"StardistSegmenter expects YX or YXC image for 2D. Got shape: {image.shape}"
                )

        # Convert multi-channel to grayscale if needed
        # StarDist models expect single channel (grayscale)
        if is_3d_model and image.ndim == 4 and image.shape[-1] in [3, 4]:
            print(
                f"⚠️  Converting {image.shape[-1]}-channel 3D image to grayscale for StarDist"
            )
            x = np.mean(image, axis=-1)
        elif not is_3d_model and image.ndim == 3 and image.shape[-1] in [3, 4]:
            print(
                f"⚠️  Converting {image.shape[-1]}-channel 2D image to grayscale for StarDist"
            )
            x = np.mean(image, axis=-1)
        else:
            x = image

        # Optional normalization
        x = self._normalize(x) if self.normalize_input else x

        print(
            f"🔍 StarDist input: shape={x.shape}, dtype={x.dtype}, ndim={x.ndim}"
        )

        # Load model if not already loaded
        if model is None:
            try:
                if is_3d_model:
                    model = StarDist3D.from_pretrained(self.model_preset)
                    print(
                        f"Loaded StarDist3D pretrained model: {self.model_preset}"
                    )
                else:
                    model = StarDist2D.from_pretrained(self.model_preset)
                    print(
                        f"Loaded StarDist2D pretrained model: {self.model_preset}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Could not load StarDist model '{self.model_preset}': {e}"
                ) from e

        # Predict instances
        try:
            # Check TensorFlow GPU availability
            print("🔍 TensorFlow GPU Check:")
            try:
                import tensorflow as tf

                print(f"   TensorFlow version: {tf.__version__}")
                print(
                    f"   GPU available: {tf.config.list_physical_devices('GPU')}"
                )
                print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
                if tf.config.list_physical_devices("GPU"):
                    print("   ✅ GPU ENABLED - using GPU acceleration")
                else:
                    print("   ⚠️  NO GPU - using CPU (will be slow!)")
            except (ImportError, AttributeError, RuntimeError) as e:
                print(f"   ❌ Could not check TensorFlow GPU: {e}")

            # Set axes based on model type
            if is_3d_model:
                axes = "ZYX"
                print(
                    f"🔍 Calling model.predict_instances with shape={x.shape}, axes={axes}"
                )
            else:
                axes = None  # Let 2D model use default
                print(
                    f"🔍 Calling model.predict_instances with shape={x.shape}, axes=None (will use model default)"
                )

            labels, details = model.predict_instances(
                x,
                axes=axes,
                prob_thresh=self.prob_thresh,
                nms_thresh=self.nms_thresh,
            )
            print(
                f"StarDist: Found {len(np.unique(labels)) - 1} objects (prob_thresh={self.prob_thresh}, nms_thresh={self.nms_thresh})"
            )
        except Exception as e:
            print("❌ StarDist predict_instances failed!")
            print(f"   Input shape: {x.shape}")
            print(f"   Input dtype: {x.dtype}")
            print(f"   Model preset: {self.model_preset}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback

            print("   Full traceback:")
            traceback.print_exception(type(e), e, e.__traceback__)
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

        execution_code = f"""
import numpy as np
from stardist.models import StarDist2D

# Parameters from segmenter
model_preset = "{self.model_preset}"
prob_thresh = {self.prob_thresh}
nms_thresh = {self.nms_thresh}

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
print('Executing StarDist...')
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
