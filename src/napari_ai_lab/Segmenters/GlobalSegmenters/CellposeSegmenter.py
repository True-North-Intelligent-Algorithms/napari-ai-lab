"""
Cellpose Global Segmenter.

This module provides a Cellpose segmenter for automatic cell segmentation
of entire images without user prompts.
"""

import json
import os
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


# Model to axis mapping - defines the expected input axes for each Cellpose v3 model
BUILTIN_MODEL_MAP_CP3 = {
    "cyto3": "YX",  # Cellpose 3 cytoplasm model
}

# Model to axis mapping - defines the expected input axes for CellposeSAM models
BUILTIN_MODEL_MAP_CPSAM = {
    "cpsam": "YX",  # CellposeSAM model
}


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

    diameter: float = field(
        default=30.0,
        metadata={
            "type": "float",
            "param_type": "inference",
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
            "param_type": "inference",
            "default": True,
        },
    )

    # Probability threshold (cell probability)
    prob_threshold: float = field(
        default=0.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": -5,
            "max": 5,
            "step": 0.2,
            "default": 0.0,
        },
    )

    flow_threshold: float = field(
        default=0.4,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": -5,
            "max": 5,
            "step": 0.2,
            "default": 0.4,
        },
    )

    cellpose_iterations: int = field(
        default=200,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 50,
            "max": 5000,
            "step": 50,
            "default": 200,
        },
    )

    # Training parameters
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

    learning_rate: float = field(
        default=0.0001,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
            "default": 0.0001,
        },
    )

    weight_decay: float = field(
        default=0.0001,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
            "default": 0.0001,
        },
    )

    nimg_per_epoch: int = field(
        default=50,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 1000,
            "step": 10,
            "default": 50,
        },
    )

    bsize: int = field(
        default=8,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 64,
            "step": 1,
            "default": 8,
        },
    )

    rescale: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "training",
            "default": True,
        },
    )

    chan_segment: int = field(
        default=0,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 0,
            "max": 3,
            "step": 1,
            "default": 0,
        },
    )

    chan2: int = field(
        default=0,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 0,
            "max": 3,
            "step": 1,
            "default": 0,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

        # Set supported axes
        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC", "TYX", "TYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC", "TYX", "TYXC"]

        # Training paths (set by nd_easy_segment before calling train())
        self.patch_path = ""
        self.model_save_dir = ""
        self.training_model_name = ""

        # Set default inference model based on cellpose version
        if _is_cellpose_available:
            major_version = int(_cellpose_major_version)
            if major_version < 4:
                self.inference_model_name = "cyto3"  # Cellpose 3 default
            else:
                self.inference_model_name = (
                    "cpsam"  # Cellpose 4 default (CellposeSAM)
                )
        else:
            self.inference_model_name = "cyto2"  # Fallback

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

    def get_model_axis_map(self) -> dict:
        """
        Get the complete model-to-axis mapping (builtins + pretrained).

        Returns:
            dict: Dictionary mapping model names to recommended axes.
        """
        # Check cellpose version to determine which builtin map to use
        version = self.get_version()
        if version and _is_cellpose_available:
            major_version = int(_cellpose_major_version)
            if major_version < 4:
                result = BUILTIN_MODEL_MAP_CP3.copy()
            else:
                result = BUILTIN_MODEL_MAP_CPSAM.copy()
        else:
            result = {}

        # Add pretrained models from project directory
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """
        Scan ``model_save_dir`` for user-trained Cellpose models.

        A subdirectory is recognised as containing a Cellpose model when
        it contains .pth files (Cellpose model files). The subdirectory
        name becomes the model name, and we default to "YX" axis since
        Cellpose doesn't store axis information in a config file.

        Returns:
            dict: {model_name: axis_string} for every valid model found.
        """
        if not self.model_save_dir or not os.path.isdir(self.model_save_dir):
            return {}

        cellpose_save_dir = os.path.join(self.model_save_dir, "models")
        if not os.path.isdir(cellpose_save_dir):
            return {}

        pretrained = {}
        for entry in os.listdir(cellpose_save_dir):
            pretrained[entry] = "YX"

        return pretrained

    def set_model(self, model_name):
        """Set the inference model name when user selects from dropdown.

        Args:
            model_name: Name of the model to use for inference.
        """
        self.inference_model_name = model_name

        # Determine model source for logging
        if model_name in BUILTIN_MODEL_MAP_CP3:
            print(f"🔄 Selected Cellpose 3 model: {model_name}")
        elif model_name in BUILTIN_MODEL_MAP_CPSAM:
            print(f"🔄 Selected CellposeSAM model: {model_name}")
        elif model_name in self.build_pretrained_model_map():
            print(f"🔄 Selected user-trained model: {model_name}")
        else:
            print(f"🔄 Selected model: {model_name}")

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
                if self.inference_model_name in BUILTIN_MODEL_MAP_CPSAM:
                    print(
                        f"Using CellposeSAM model: {self.inference_model_name}"
                    )
                    model = models.CellposeModel(
                        gpu=self.use_gpu, model_type=self.inference_model_name
                    )
                else:
                    full_model_name = os.path.join(
                        self.model_save_dir,
                        "models",
                        self.inference_model_name,
                    )
                    model = models.CellposeModel(
                        gpu=self.use_gpu,
                        pretrained_model=full_model_name,
                    )

            else:
                if self.inference_model_name in BUILTIN_MODEL_MAP_CP3:
                    print(
                        f"Using Cellpose v3 model: {self.inference_model_name}"
                    )
                    model = models.Cellpose(
                        gpu=self.use_gpu, model_type=self.inference_model_name
                    )
                else:
                    full_model_name = os.path.join(
                        self.model_save_dir,
                        "models",
                        self.inference_model_name,
                    )
                    model = models.Cellpose(
                        gpu=self.use_gpu,
                        pretrained_model=full_model_name,
                    )
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Error creating model with GPU, falling back to CPU: {e}")
            if _cellpose_major_version == "4":
                model = models.CellposeModel(
                    gpu=False, model_type=self.inference_model_name
                )
            else:
                model = models.Cellpose(
                    gpu=False, model_type=self.inference_model_name
                )

        if image.dtype == np.int32:
            image = image.astype(np.float32)

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

    def train(self, updater=None):
        """
        Train a Cellpose model on pre-generated patches.

        Reads info.json from ``self.patch_path`` to determine axes, collects
        training data, splits into train/test, and trains a new Cellpose
        model.  The model is saved under ``self.model_save_dir / self.training_model_name``.

        This signature mirrors StardistSegmenter.train() so that
        ``_run_training`` can call every segmenter the same way.

        Args:
            updater: Optional callable ``updater(epoch, total_epochs, message)``
                     for reporting progress back to the UI.

        Returns:
            dict with keys ``success`` (bool) and ``message`` (str).
        """
        from cellpose import models, train

        from ...utilities.dl_util import (
            collect_training_data,
            divide_training_data,
        )

        # ---- resolve paths from self (set by _run_training) ----
        patch_path = self.patch_path
        model_name = self.training_model_name
        model_base_path = self.model_save_dir

        if not patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not model_base_path:
            return {"success": False, "message": "model_save_dir is not set."}
        if not model_name:
            model_name = "cellpose_model"

        # ---- read axes from info.json ----
        json_path = os.path.join(patch_path, "info.json")
        with open(json_path) as f:
            info = json.load(f)
        axes = info["axes"]

        # Determine if we need channels
        if axes == "YXC":
            n_channel_in = 3
            add_trivial_channel = False
        else:
            n_channel_in = 1
            add_trivial_channel = True  # Cellpose expects channels dim

        # ---- collect & split data ----
        X, Y = collect_training_data(
            patch_path,
            normalize_input=False,
            add_trivial_channel=add_trivial_channel,
        )

        X_train, Y_train, X_test, Y_test = divide_training_data(
            X, Y, val_size=2
        )

        # Convert to numpy arrays
        X_train = np.array(X_train).astype(np.float32)
        Y_train = np.array(Y_train).astype(np.uint16)
        X_test = np.array(X_test).astype(np.float32)
        Y_test = np.array(Y_test).astype(np.uint16)

        # Get cellpose major version
        major_number = int(_cellpose_major_version)

        msg = (
            f"🏋️ Training Cellpose v{cellpose.version}: {len(X_train)} train, {len(X_test)} test\n"
            f"   axes={axes}, n_channel_in={n_channel_in}\n"
            f"   epochs={self.num_epochs}, batch_size={self.bsize}\n"
            f"   learning_rate={self.learning_rate}, weight_decay={self.weight_decay}"
        )
        print(msg)
        if updater is not None:
            updater(0, self.num_epochs, msg)

        # ---- create model ----
        # TODO: Training start point will be current inference model but consider
        # making dropdown to choose training start point model
        try:
            if major_number >= 4:
                if self.inference_model_name in BUILTIN_MODEL_MAP_CPSAM:
                    print(
                        f"Training from CellposeSAM model: {self.inference_model_name}"
                    )
                    model = models.CellposeModel(
                        gpu=self.use_gpu, model_type=self.inference_model_name
                    )
                else:
                    full_model_name = os.path.join(
                        self.model_save_dir,
                        "models",
                        self.inference_model_name,
                    )
                    model = models.CellposeModel(
                        gpu=self.use_gpu,
                        pretrained_model=full_model_name,
                    )
            else:
                if self.inference_model_name in BUILTIN_MODEL_MAP_CP3:
                    print(
                        f"Training from Cellpose v3 model: {self.inference_model_name}"
                    )
                    model = models.Cellpose(
                        gpu=self.use_gpu, model_type=self.inference_model_name
                    )
                else:
                    full_model_name = os.path.join(
                        self.model_save_dir,
                        "models",
                        self.inference_model_name,
                    )
                    model = models.Cellpose(
                        gpu=self.use_gpu,
                        pretrained_model=full_model_name,
                    )
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Error creating model with GPU, falling back to CPU: {e}")
            if major_number >= 4:
                model = models.CellposeModel(
                    gpu=False, model_type=self.inference_model_name
                )
            else:
                model = models.Cellpose(
                    gpu=False, model_type=self.inference_model_name
                )

        # Create save directory
        save_path = model_base_path
        os.makedirs(save_path, exist_ok=True)

        print(
            f"Training Cellpose model (version {major_number}.x) with {len(X_train)} training images..."
        )

        # ---- create progress updater wrapper ----
        class _ProgressUpdater:
            """Wrapper to track training progress."""

            def __init__(self, updater_fn, num_epochs):
                self.updater_fn = updater_fn
                self.num_epochs = num_epochs
                self.current_epoch = 0

            def update(self, epoch, loss, test_loss=None):
                """Called by train_seg after each epoch."""
                self.current_epoch = epoch
                if self.updater_fn is not None:
                    test_msg = (
                        f", test_loss: {test_loss:.4f}" if test_loss else ""
                    )
                    self.updater_fn(
                        epoch,
                        self.num_epochs,
                        f"Epoch {epoch}/{self.num_epochs} — loss: {loss:.4f}{test_msg}",
                    )

        # ---- train model with version-specific parameters ----
        try:
            if major_number < 4:
                # Cellpose 3.x - use bsize parameter
                train.train_seg(
                    model.net if hasattr(model, "net") else model.cp.net,
                    X_train,
                    Y_train,
                    test_data=X_test,
                    test_labels=Y_test,
                    channels=[self.chan_segment, self.chan2],
                    save_path=save_path,
                    n_epochs=self.num_epochs,
                    rescale=self.rescale,
                    normalize=False,
                    bsize=self.bsize,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    model_name=model_name,
                    min_train_masks=0,
                )
            else:
                # Cellpose 4.x - use nimg_per_epoch instead of bsize
                train.train_seg(
                    model.net if hasattr(model, "net") else model.cp.net,
                    X_train,
                    Y_train,
                    test_data=X_test,
                    test_labels=Y_test,
                    save_path=save_path,
                    n_epochs=self.num_epochs,
                    rescale=self.rescale,
                    normalize=False,
                    learning_rate=self.learning_rate,
                    weight_decay=self.weight_decay,
                    nimg_per_epoch=self.nimg_per_epoch,
                    model_name=model_name,
                    min_train_masks=0,
                )

            done_msg = f"✅ Training complete. Model saved to: {save_path}"
            print(done_msg)
            if updater is not None:
                updater(self.num_epochs, self.num_epochs, done_msg)

            # Set inference_model_name so UI can select the trained model
            self.inference_model_name = model_name

            return {"success": True, "message": done_msg}

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            TypeError,
            AttributeError,
        ) as e:
            error_msg = f"❌ Training failed: {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            return {"success": False, "message": error_msg}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("CellposeSegmenter", cls)
