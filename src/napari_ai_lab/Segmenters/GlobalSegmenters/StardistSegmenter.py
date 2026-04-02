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


# Model to axis mapping - defines the expected input axes for each model
BUILTIN_MODEL_MAP = {
    "2D_versatile_fluo": "YX",  # Grayscale fluorescence
    "2D_versatile_he": "YXC",  # RGB H&E staining
    "3D_demo": "ZYX",  # 3D grayscale
}


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
            "choices": list(BUILTIN_MODEL_MAP.keys()),
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

    num_epochs: int = field(
        default=100,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 1000,
            "step": 1,
            "default": 100,
        },
    )

    train_patch_size_y: int = field(
        default=256,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 32,
            "max": 1024,
            "step": 32,
            "default": 256,
        },
    )

    train_patch_size_x: int = field(
        default=256,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 32,
            "max": 1024,
            "step": 32,
            "default": 256,
        },
    )

    steps_per_epoch: int = field(
        default=100,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 1000,
            "step": 10,
            "default": 100,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()
        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self.custom_model = None
        self.is_3d_model = False
        # Set by nd_easy_segment before calling train()
        self.patch_path = ""
        self.model_save_dir = ""
        self.model_name = ""

    def get_recommended_axis(self) -> str:
        """
        Get the recommended axis for the current model preset.

        Returns:
            str: Recommended axis string (e.g., "YX", "YXC", "ZYX")
        """
        return BUILTIN_MODEL_MAP.get(self.model_preset, "YX")

    @staticmethod
    def get_model_axis_map() -> dict:
        """
        Get the complete model-to-axis mapping.

        Returns:
            dict: Dictionary mapping model names to recommended axes
        """
        return BUILTIN_MODEL_MAP.copy()

    def __setattr__(self, name, value):
        """Override setattr to detect model_path and model_preset changes."""
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

        # Check if model_preset changed
        elif (
            name == "model_preset"
            and old_value != value
            and old_value is not None
        ):
            recommended_axis = self.get_recommended_axis()
            print(
                f"🔄 Model preset changed to '{value}' - recommended axis: {recommended_axis}"
            )

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

    def train(self, updater=None):
        """
        Train a StarDist2D model on pre-generated patches.

        Reads info.json from ``self.patch_path`` to determine axes, collects
        training data, splits into train/val, and trains a new StarDist2D
        model.  The model is saved under ``self.model_save_dir / self.model_name``.

        This signature mirrors MonaiUNetSegmenter.train() so that
        ``_run_training`` can call every segmenter the same way.

        Args:
            updater: Optional callable ``updater(message, progress_percent)``
                     for reporting progress back to the UI.  It is wrapped
                     in a Keras callback internally.

        Returns:
            dict with keys ``success`` (bool) and ``message`` (str).
        """
        import json

        import keras
        from stardist.models import Config2D, StarDist2D

        from ...utilities.dl_util import (
            collect_training_data,
            divide_training_data,
        )

        # ---- resolve paths from self (set by _run_training) ----
        patch_path = self.patch_path
        model_name = self.model_name
        model_base_path = self.model_save_dir

        if not patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not model_base_path:
            return {"success": False, "message": "model_save_dir is not set."}
        if not model_name:
            model_name = "stardist_model"

        # ---- read axes from info.json ----
        json_path = os.path.join(patch_path, "info.json")
        with open(json_path) as f:
            info = json.load(f)
        axes = info["axes"]

        if axes == "YXC":
            n_channel_in = 3
            add_trivial_channel = False
        else:
            n_channel_in = 1
            add_trivial_channel = True

        # ---- collect & split data ----
        X, Y = collect_training_data(
            patch_path,
            normalize_input=False,
            add_trivial_channel=add_trivial_channel,
        )
        X_train, Y_train, X_val, Y_val = divide_training_data(X, Y, val_size=2)

        msg = (
            f"🏋️ Training StarDist2D: {len(X_train)} train, {len(X_val)} val\n"
            f"   axes={axes}, n_channel_in={n_channel_in}\n"
            f"   epochs={self.num_epochs}, steps_per_epoch={self.steps_per_epoch}\n"
            f"   train_patch_size=({self.train_patch_size_y}, {self.train_patch_size_x})"
        )
        print(msg)
        if updater is not None:
            updater(0, self.num_epochs, msg)

        # ---- build Keras callback that wraps the updater ----
        class _ProgressCallback(keras.callbacks.Callback):
            """Relay Keras epoch events to the napari-ai-lab updater."""

            def __init__(self, updater_fn, num_epochs):
                super().__init__()
                self._updater = updater_fn
                self._num_epochs = num_epochs

            def on_epoch_begin(self, epoch, logs=None):
                if self._updater is not None:
                    self._updater(
                        epoch,
                        self._num_epochs,
                        f"Starting epoch {epoch + 1}/{self._num_epochs}",
                    )

            def on_epoch_end(self, epoch, logs=None):
                if self._updater is not None:
                    loss = (logs or {}).get("loss", float("nan"))
                    val_loss = (logs or {}).get("val_loss", float("nan"))
                    self._updater(
                        epoch,
                        self._num_epochs,
                        f"Epoch {epoch + 1}/{self._num_epochs} — "
                        f"loss: {loss:.4f}, val_loss: {val_loss:.4f}",
                    )

        custom_callback = _ProgressCallback(updater, self.num_epochs)

        # ---- create model & train ----
        config = Config2D(
            n_rays=32,
            axes=axes,
            n_channel_in=n_channel_in,
            train_patch_size=(
                self.train_patch_size_y,
                self.train_patch_size_x,
            ),
            unet_n_depth=3,
        )
        model = StarDist2D(
            config=config, name=model_name, basedir=model_base_path
        )
        model.prepare_for_training()

        if custom_callback is not None:
            custom_callback.num_epochs = self.num_epochs
            model.callbacks.append(custom_callback)

        model.train(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        # ---- store trained model for immediate use ----
        self.custom_model = model
        self.model_path = os.path.join(model_base_path, model_name)
        self.model_file_path = self.model_path
        done_msg = f"✅ Training complete. Model saved to: {self.model_path}"
        print(done_msg)
        if updater is not None:
            updater(done_msg, 100)

        return {"success": True, "message": done_msg}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("StardistSegmenter", cls)
