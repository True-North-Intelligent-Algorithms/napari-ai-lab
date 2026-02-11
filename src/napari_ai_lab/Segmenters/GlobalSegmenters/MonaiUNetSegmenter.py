"""
MONAI UNet Global Segmenter.

This module provides a MONAI UNet segmenter for automatic semantic segmentation
of entire images without user prompts.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import MONAI dependencies
try:
    from monai.inferers import sliding_window_inference

    _is_monai_available = True
except ImportError:
    _is_monai_available = False


@dataclass
class MonaiUNetSegmenter(GlobalSegmenterBase):
    """
    MONAI UNet global segmenter.

    This segmenter uses a trained MONAI UNet model for automatic semantic
    segmentation on entire images using sliding window inference.
    """

    instructions = """
MONAI UNet Automatic Segmentation:
• Uses trained UNet model for semantic segmentation
• Tile Size: Size of patches for sliding window inference
• Show Background Class: Include background in output
• Requires pre-trained model loaded from disk
• Best for: semantic segmentation with custom trained models
• Works with 2D and 3D images
    """

    # Parameters for MONAI UNet segmentation
    tile_size: int = field(
        default=1024,
        metadata={
            "type": "int",
            "min": 128,
            "max": 100000,
            "step": 128,
            "default": 1024,
        },
    )

    show_background_class: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "default": True,
        },
    )

    model_name: str = field(
        default="",
        metadata={
            "type": "file",
            "file_type": "file",
            "default": "",
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

        # Model will be loaded from disk
        self.model = None

        # Set supported axes
        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]

    def __setattr__(self, name, value):
        """Override setattr to detect model_name changes."""
        # Get old value if it exists
        old_value = getattr(self, name, None) if hasattr(self, name) else None

        # Set the new value
        super().__setattr__(name, value)

        # Check if this is model_name and value changed
        if name == "model_name" and old_value != value and value:
            print(f"🔄 Model name changed from '{old_value}' to '{value}'")
            self._on_model_name_changed(value)

    def _on_model_name_changed(self, model_path: str):
        """Handle model name changes - load the PyTorch model."""
        print(f"📁 Loading MONAI UNet model from: {model_path}")

        try:
            self.model = torch.load(model_path, weights_only=False)
            print(
                f"✅ Successfully loaded MONAI UNet model from: {model_path}"
            )
            print(f"   Model type: {type(self.model)}")
        except Exception as e:  # noqa
            print(f"❌ Failed to load model from {model_path}: {e}")
            self.model = None

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if MONAI and torch can be imported, False otherwise.
        """
        return _is_monai_available

    def get_version(self):
        """Get the MONAI version."""
        if _is_monai_available:
            try:
                import monai

                return f"monai-{monai.__version__}"
            except (ImportError, AttributeError):
                return "monai-unknown"
        return None

    def load_model(self, model_path: str):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model file (.pth)
        """
        if not _is_monai_available:
            raise ImportError("MONAI is not available. Cannot load model.")

        # Setting model_name will trigger __setattr__ which loads the model
        self.model_name = model_path

    def segment(self, image, **kwargs):
        """
        Perform MONAI UNet segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Labeled segmentation mask.

        Raises:
            ValueError: If image dimensions are not supported or model not loaded.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model() before segmentation."
            )

        if len(image.shape) < 2:
            raise ValueError(
                f"MonaiUNetSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Prepare image for inference
        image_norm = self._normalize_image(image)

        # Prepare tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Add channel and batch dimensions if needed
        if len(image_norm.shape) == 2:  # YX -> 1, 1, Y, X
            image_tensor = torch.from_numpy(image_norm[None, None, :, :])
        elif len(image_norm.shape) == 3:  # ZYX or YXC
            # Assume ZYX for 3D
            image_tensor = torch.from_numpy(image_norm[None, None, :, :, :])
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image_tensor = image_tensor.float().to(device)

        # Perform sliding window inference
        with torch.no_grad():
            if len(image.shape) == 2:
                roi_size = (self.tile_size, self.tile_size)
            else:
                roi_size = (self.tile_size, self.tile_size)

            y = sliding_window_inference(
                image_tensor,
                roi_size,
                sw_batch_size=1,
                predictor=self.model,
                mode="gaussian",
                overlap=0.125,
            )

        # Apply softmax and get predicted classes
        probabilities = F.softmax(y, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        # Handle background class
        result = predicted_classes.cpu().numpy().squeeze()
        if not self.show_background_class:
            result = result - 1

        return (result + 1).astype(np.uint16)

    def _normalize_image(self, image):
        """
        Normalize image using quantile normalization.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Normalized image
        """
        image_float = image.astype(np.float32)

        # Simple percentile normalization
        p_low, p_high = np.percentile(image_float, [1, 99])
        if p_high > p_low:
            image_norm = (image_float - p_low) / (p_high - p_low)
            image_norm = np.clip(image_norm, 0, 1)
        else:
            image_norm = image_float

        return image_norm

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework(
            "MonaiUNetSegmenter", cls
        )

    def get_execution_string(self, image, **kwargs):
        """
        Generate execution string for remote processing.

        Args:
            image (numpy.ndarray): Input image
            **kwargs: Additional keyword arguments

        Returns:
            str: Python code string for remote execution
        """
        return f"""
# MONAI UNet Segmentation
import torch
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

# Load model
model = torch.load('{self.model_name}', weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Normalize and prepare image
image_norm = (image - image.min()) / (image.max() - image.min())
image_tensor = torch.from_numpy(image_norm[None, None, :, :]).float().to(device)

# Inference
with torch.no_grad():
    y = sliding_window_inference(
        image_tensor, ({self.tile_size}, {self.tile_size}),
        sw_batch_size=1, predictor=model, mode='gaussian', overlap=0.125
    )

probabilities = F.softmax(y, dim=1)
result = torch.argmax(probabilities, dim=1).cpu().numpy().squeeze()
"""
