"""
StarDist Global Segmenter.

This module provides a StarDist segmenter for automatic segmentation
of entire 2D images without user prompts.
"""

import os
from dataclasses import dataclass, field

import numpy as np

from ...utilities.dl_util import normalize_image
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

    # model_preset is NOT a dataclass field — it's set as a plain attribute
    # in __post_init__ and managed by the model_preset_combo in nd_easy_segment.
    # This avoids the form trying to create an unsupported widget for it.

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

    # model_path removed — model selection is done via model_preset_combo
    # in nd_easy_segment, populated by get_model_axis_map() which includes
    # both BUILTIN_MODEL_MAP entries and user-trained models from model_save_dir.

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
        self.models = {}
        # Set by nd_easy_segment before calling train() or segment()
        self.patch_path = ""
        self.model_save_dir = ""
        self.model_name = ""
        # Model preset: selected via combo in nd_easy_segment
        self.model_preset = "2D_versatile_fluo"

    def get_recommended_axis(self) -> str:
        """
        Get the recommended axis for the current model preset.

        Returns:
            str: Recommended axis string (e.g., "YX", "YXC", "ZYX")
        """
        full_map = self.get_model_axis_map()
        return full_map.get(self.model_preset, "YX")

    def get_model_axis_map(self) -> dict:
        """
        Get the complete model-to-axis mapping (builtins + pretrained).

        Returns:
            dict: Dictionary mapping model names to recommended axes.
        """
        result = BUILTIN_MODEL_MAP.copy()
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """
        Scan ``model_save_dir`` for user-trained StarDist models.

        A subdirectory is recognised as a model when it contains a
        ``config.json`` with at least ``axes`` and ``n_channel_in``.
        The axis string is cleaned: if ``n_channel_in == 1`` the trailing
        ``C`` (if present) is dropped.

        Returns:
            dict: {model_name: axis_string} for every valid model found.
        """
        import json

        if not self.model_save_dir or not os.path.isdir(self.model_save_dir):
            return {}

        pretrained = {}
        for entry in os.listdir(self.model_save_dir):
            model_dir = os.path.join(self.model_save_dir, entry)
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.isfile(config_path):
                continue
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                axes = cfg.get("axes", "YX")
                n_channel_in = cfg.get("n_channel_in", 1)
                # Drop trailing C when the model expects single-channel input
                if n_channel_in == 1 and axes.endswith("C"):
                    axes = axes[:-1]
                pretrained[entry] = axes
            except (json.JSONDecodeError, OSError):
                continue
        return pretrained

    def set_model(self, model_name):
        """Load the model for model_name and cache it in self.models dict."""
        self.model_preset = model_name

        # Already cached — nothing to do
        if model_name in self.models:
            print(f"🔄 Model '{model_name}' already loaded, reusing cached.")
            return

        model_axis = self.get_model_axis_map().get(model_name, "YX")
        is_3d = "Z" in model_axis
        is_user_trained = model_name in self.build_pretrained_model_map()

        if is_user_trained:
            if is_3d:
                model = StarDist3D(
                    config=None, name=model_name, basedir=self.model_save_dir
                )
            else:
                model = StarDist2D(
                    config=None, name=model_name, basedir=self.model_save_dir
                )
            print(
                f"Loaded user-trained model: {model_name} from {self.model_save_dir}"
            )
        elif is_3d:
            model = StarDist3D.from_pretrained(model_name)
            print(f"Loaded builtin StarDist3D: {model_name}")
        else:
            model = StarDist2D.from_pretrained(model_name)
            print(f"Loaded builtin StarDist2D: {model_name}")

        self.models[model_name] = model

    def __setattr__(self, name, value):
        """Override setattr to detect model_preset changes."""
        # Get old value if it exists
        old_value = getattr(self, name, None) if hasattr(self, name) else None

        # Set the new value
        super().__setattr__(name, value)

        # Check if model_preset changed
        if (
            name == "model_preset"
            and old_value != value
            and old_value is not None
        ):
            recommended_axis = self.get_recommended_axis()
            print(
                f"🔄 Model preset changed to '{value}' - recommended axis: {recommended_axis}"
            )

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if stardist can be imported, False otherwise.
        """
        return _is_stardist_available

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using percentile normalization (p1=1, p99=99) from dl_util."""
        return normalize_image(image, intensity_low=1, p_high=99)

    def segment(self, image, **kwargs):
        """
        Perform StarDist segmentation on 2D or 3D image.

        Args:
            image (numpy.ndarray): Input image (YX, YXC, ZYX, or ZYXC).

        Returns:
            numpy.ndarray: Labeled segmentation mask.
        """
        # Load model on first use if not yet cached
        if self.model_preset not in self.models:
            self.set_model(self.model_preset)

        model = self.models[self.model_preset]
        model_axis = self.get_model_axis_map().get(self.model_preset, "YX")
        is_3d_model = "Z" in model_axis
        print(f"Using model: {self.model_preset} (axis: {model_axis})")

        # Convert multi-channel to grayscale if needed
        if is_3d_model and image.ndim == 4 and image.shape[-1] in [3, 4]:
            print(
                f"⚠️  Converting {image.shape[-1]}-channel 3D image to grayscale for StarDist"
            )
            x = np.mean(image, axis=-1)
        elif (
            not is_3d_model
            and image.ndim == 3
            and image.shape[-1] in [3, 4]
            and model.config.n_channel_in == 1
        ):
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

        # ---- store trained model in cache for immediate use ----
        self.models[model_name] = model
        self.model_preset = model_name
        model_full_path = os.path.join(model_base_path, model_name)
        done_msg = f"✅ Training complete. Model saved to: {model_full_path}"
        print(done_msg)
        if updater is not None:
            updater(self.num_epochs, self.num_epochs, msg)

        return {"success": True, "message": done_msg}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("StardistSegmenter", cls)
