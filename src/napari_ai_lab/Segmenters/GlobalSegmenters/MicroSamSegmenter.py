"""
MicroSam Global Segmenter.

This module provides a MicroSam segmenter for automatic instance segmentation
of entire images without user prompts.
"""

import glob
import os
import shutil
import tempfile
from dataclasses import dataclass, field

import numpy as np

from ...utility import get_ndim
from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import micro_sam at module level
try:
    import micro_sam
    import torch
    from micro_sam import training as sam_training
    from micro_sam.automatic_segmentation import (
        automatic_instance_segmentation,
        get_predictor_and_segmenter,
    )
    from torch_em.data import MinInstanceSampler

    _is_microsam_available = True
except ImportError:
    micro_sam = None
    sam_training = None
    _is_microsam_available = False

# Try to import imageio for image conversion
try:
    import imageio

    _is_imageio_available = True
except ImportError:
    imageio = None
    _is_imageio_available = False


# Model to axis mapping - defines the expected input axes for each MicroSAM model
BUILTIN_MODEL_MAP = {
    "vit_b": "YX",  # Vision Transformer Base
    "vit_l": "YX",  # Vision Transformer Large
    "vit_h": "YX",  # Vision Transformer Huge
    "vit_b_lm": "YX",  # Vision Transformer Base - Light Microscopy
    "vit_t": "YX",  # Vision Transformer Tiny
}


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
    # Note: model_type is kept for internal use but model selection is via dropdown (inference_model_name)
    model_type: str = field(
        default="vit_b_lm",
        metadata={
            "type": "str",
            "param_type": "internal",  # Not shown in UI - controlled by model dropdown
            "default": "vit_b_lm",
        },
    )

    use_tiling: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "default": False,
        },
    )

    tile_width: int = field(
        default=384,
        metadata={
            "type": "int",
            "param_type": "inference",
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
            "param_type": "inference",
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
            "param_type": "inference",
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
            "param_type": "inference",
            "min": 0,
            "max": 256,
            "step": 16,
            "default": 64,
        },
    )

    # foreground threshold, lower to find more objects
    foreground_threshold: float = field(
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

    # center_distance_threshold controls the threshold for creating seeds
    # from the (normalized) distance to the center predicted by the model
    # high values lead to more segmented objects
    # low values connect object segments together
    center_distance_threshold: float = field(
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

    # boundary_distance_threshold control the threshold for creating seeds
    # from the (normalized, inverted) distances predicuted by the model.
    # high values lead to more segmented objects
    # low values connect object segments together
    boundary_distance_threshold: float = field(
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

    process_channels_separate: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "default": False,
        },
    )

    channel_to_process: int = field(
        default=0,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 10,
            "step": 1,
            "default": 0,
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
            "step": 10,
            "default": 100,
        },
    )

    n_objects_per_batch: int = field(
        default=2,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 16,
            "step": 1,
            "default": 2,
        },
    )

    batch_size: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 8,
            "step": 1,
            "default": 1,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()

        # Set supported axes
        self._supported_axes = ["YX", "ZYX"]
        self._potential_axes = ["YX", "ZYX"]

        # Training paths (set by nd_easy_segment before calling train())
        self.patch_path = ""
        self.model_save_dir = ""
        self.training_model_name = ""

        # Set default inference model
        self.inference_model_name = "vit_b_lm"

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

    def get_model_axis_map(self) -> dict:
        """
        Get the complete model-to-axis mapping (builtins + pretrained).

        Returns:
            dict: Dictionary mapping model names to recommended axes.
        """
        result = BUILTIN_MODEL_MAP.copy()
        # Add pretrained models from project directory
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """
        Scan ``model_save_dir/checkpoints`` for user-trained MicroSAM models.

        MicroSAM stores trained models in checkpoints/<model_name>/best.pt.
        Each subdirectory under checkpoints/ is recognized as a model.

        Returns:
            dict: {model_name: axis_string} for every valid model found.
        """
        if not self.model_save_dir or not os.path.isdir(self.model_save_dir):
            return {}

        checkpoints_dir = os.path.join(self.model_save_dir, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            return {}

        pretrained = {}
        for entry in os.listdir(checkpoints_dir):
            model_dir = os.path.join(checkpoints_dir, entry)
            if not os.path.isdir(model_dir):
                continue

            # Check if directory contains best.pt file
            best_pt = os.path.join(model_dir, "best.pt")
            if os.path.isfile(best_pt):
                # Default to YX axis for MicroSAM models
                pretrained[entry] = "YX"

        return pretrained

    def set_model(self, model_name):
        """Set the inference model name when user selects from dropdown.

        Args:
            model_name: Name of the model to use for inference.
        """
        self.inference_model_name = model_name

        # Update internal model_type if it's a builtin model
        if model_name in BUILTIN_MODEL_MAP:
            self.model_type = model_name
            print(f"🔄 Selected MicroSAM model: {model_name}")
        elif model_name in self.build_pretrained_model_map():
            print(f"🔄 Selected user-trained model: {model_name}")
        else:
            print(f"🔄 Selected model: {model_name}")

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

        # Determine if using builtin or custom model
        is_builtin = self.inference_model_name in BUILTIN_MODEL_MAP
        checkpoint = None

        if is_builtin:
            model_type = self.inference_model_name
            print(f"MicroSam: Loading builtin model: {model_type}")
        else:
            # Custom trained model - construct checkpoint path
            model_type = (
                self.model_type
            )  # Use base model type for custom models
            checkpoint = os.path.join(
                self.model_save_dir,
                "checkpoints",
                self.inference_model_name,
                "best.pt",
            )
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    f"Custom model checkpoint not found: {checkpoint}"
                )
            print(f"MicroSam: Loading custom model from: {checkpoint}")

        # Get predictor and segmenter
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            is_tiled=(tile_shape is not None),
        )

        if self.process_channels_separate and len(image.shape) > 2:
            image_ = np.transpose(
                image, (2, 0, 1)
            )  # transpose to (C, H, W) format
            ndim = None
        else:
            image_ = image
            ndim = get_ndim(image.shape)

        # Perform automatic instance segmentation
        print(f"MicroSam: Running segmentation (ndim={ndim})...")
        result = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image_,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
            foreground_threshold=self.foreground_threshold,
            center_distance_threshold=self.center_distance_threshold,
            boundary_distance_threshold=self.boundary_distance_threshold,
        )

        print(f"MicroSam: Found {len(np.unique(result)) - 1} instances")

        if self.process_channels_separate and len(image.shape) > 2:
            return result[self.channel_to_process]
        else:
            return result.astype(np.uint16)

    def _generate_execution_string(self, image):
        """Generate execution string for remote processing."""
        tile_shape_str = (
            f"({self.tile_height}, {self.tile_width})"
            if self.use_tiling
            else "None"
        )

        # Determine model type and checkpoint for remote execution
        is_builtin = self.inference_model_name in BUILTIN_MODEL_MAP
        if is_builtin:
            model_type_str = f'"{self.inference_model_name}"'
            checkpoint_str = "None"
        else:
            model_type_str = f'"{self.model_type}"'
            checkpoint_str = f'"{os.path.join(self.model_save_dir, "checkpoints", self.inference_model_name, "best.pt")}"'

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
    model_type = {model_type_str}
    checkpoint = {checkpoint_str}
    foreground_threshold = {self.foreground_threshold}
    center_distance_threshold = {self.center_distance_threshold}
    boundary_distance_threshold = {self.boundary_distance_threshold}

    # Get predictor and segmenter
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint,
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
        foreground_threshold=foreground_threshold,
        center_distance_threshold=center_distance_threshold,
        boundary_distance_threshold=boundary_distance_threshold,
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

    def train(self, updater=None):
        """
        Train a MicroSAM model on pre-generated patches.

        Reads patches from ``self.patch_path``, converts to 8-bit if necessary,
        and trains a new MicroSAM model. The model is saved under
        ``self.model_save_dir / self.training_model_name``.

        MicroSAM requires 8-bit uint8 images for training. This method
        automatically converts patches to 8-bit format in a temporary directory.

        Args:
            updater: Optional callable ``updater(epoch, total_epochs, message)``
                     for reporting progress back to the UI.

        Returns:
            dict with keys ``success`` (bool) and ``message`` (str).
        """
        if not _is_microsam_available:
            return {
                "success": False,
                "message": "MicroSAM is not available. Please install micro_sam.",
            }

        if not _is_imageio_available:
            return {
                "success": False,
                "message": "imageio is not available. Please install imageio.",
            }

        # ---- resolve paths from self (set by _run_training) ----
        patch_path = self.patch_path
        model_name = self.training_model_name
        model_save_dir = self.model_save_dir

        if not patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not model_save_dir:
            return {"success": False, "message": "model_save_dir is not set."}
        if not model_name:
            model_name = "microsam_model"

        # ---- locate input patches ----
        input_dir = os.path.join(patch_path, "input0")
        if not os.path.isdir(input_dir):
            return {
                "success": False,
                "message": f"Input directory not found: {input_dir}",
            }

        segmentation_dir = os.path.join(patch_path, "ground_truth0")
        if not os.path.isdir(segmentation_dir):
            return {
                "success": False,
                "message": f"Ground truth directory not found: {segmentation_dir}",
            }

        msg = (
            f"🏋️ Training MicroSAM model: {model_name}\n"
            f"   Model type: {self.model_type}\n"
            f"   Epochs: {self.num_epochs}\n"
            f"   Objects per batch: {self.n_objects_per_batch}\n"
            f"   Batch size: {self.batch_size}"
        )
        print(msg)
        if updater is not None:
            updater(0, self.num_epochs, msg)

        # ---- create temporary directory for 8-bit images ----
        temp_dir_8bit = tempfile.mkdtemp(dir=patch_path, prefix="temp_8bit_")

        try:
            # ---- convert images to 8-bit ----
            image_paths = sorted(glob.glob(os.path.join(input_dir, "*")))
            if not image_paths:
                return {
                    "success": False,
                    "message": f"No images found in {input_dir}",
                }

            print(f"Converting {len(image_paths)} images to 8-bit format...")
            patch_shape = None

            for image_path in image_paths:
                image = imageio.imread(image_path)

                # Intelligent 8-bit conversion
                if image.dtype == np.uint8:
                    # Already 8-bit
                    image_8bit = image
                elif image.dtype in (np.float32, np.float64):
                    # Floating point - check range
                    if image.max() <= 1.0:
                        # Assumed normalized [0, 1] range
                        image_8bit = (image * 255).astype(np.uint8)
                    else:
                        # Arbitrary float range - normalize to [0, 255]
                        image_min, image_max = image.min(), image.max()
                        if image_max > image_min:
                            image_8bit = (
                                255
                                * (image - image_min)
                                / (image_max - image_min)
                            ).astype(np.uint8)
                        else:
                            image_8bit = np.zeros_like(image, dtype=np.uint8)
                else:
                    # Integer types (uint16, etc.) - scale to [0, 255]
                    image_min, image_max = image.min(), image.max()
                    if image_max > image_min:
                        image_8bit = (
                            255 * (image - image_min) / (image_max - image_min)
                        ).astype(np.uint8)
                    else:
                        image_8bit = np.zeros_like(image, dtype=np.uint8)

                # Save to temp directory
                output_path = os.path.join(
                    temp_dir_8bit, os.path.basename(image_path)
                )
                imageio.imwrite(output_path, image_8bit)

                # Determine patch shape from first image
                if patch_shape is None:
                    # patch_shape = [1, image_8bit.shape[0], image_8bit.shape[1]]
                    patch_shape = image_8bit.shape

            # ---- setup training data loaders ----
            raw_key, label_key = "*.tif", "*.tif"
            train_instance_segmentation = True
            sampler = MinInstanceSampler(min_size=25)

            train_loader = sam_training.default_sam_loader(
                raw_paths=temp_dir_8bit,
                raw_key=raw_key,
                label_paths=segmentation_dir,
                label_key=label_key,
                with_segmentation_decoder=train_instance_segmentation,
                patch_shape=patch_shape,
                batch_size=self.batch_size,
                # is_seg_dataset=True,
                rois=None,
                shuffle=True,
                raw_transform=sam_training.identity,
                sampler=sampler,
            )

            val_loader = sam_training.default_sam_loader(
                raw_paths=temp_dir_8bit,
                raw_key=raw_key,
                label_paths=segmentation_dir,
                label_key=label_key,
                with_segmentation_decoder=train_instance_segmentation,
                patch_shape=patch_shape,
                batch_size=self.batch_size,
                # is_seg_dataset=True,
                rois=None,
                shuffle=True,
                raw_transform=sam_training.identity,
                sampler=sampler,
            )

            # ---- determine device ----
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"Training with device: {device}, model_type: {self.model_type}, "
                f"n_objects_per_batch: {self.n_objects_per_batch}"
            )

            # ---- run training ----
            sam_training.train_sam(
                name=model_name,
                save_root=model_save_dir,
                model_type=self.model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=self.num_epochs,
                n_objects_per_batch=self.n_objects_per_batch,
                with_segmentation_decoder=train_instance_segmentation,
                # ndim = ndim,
                device=device,
            )

            model_path = os.path.join(model_save_dir, model_name)
            done_msg = f"✅ Training complete. Model saved to: {model_path}"
            print(done_msg)
            if updater is not None:
                updater(self.num_epochs, self.num_epochs, done_msg)

            # Set inference_model_name so UI can select the trained model
            self.inference_model_name = model_name

            return {"success": True, "message": done_msg}

        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as e:
            error_msg = f"❌ Training failed: {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            return {"success": False, "message": error_msg}

        finally:
            # Clean up temporary 8-bit directory
            if os.path.exists(temp_dir_8bit):
                print(f"Cleaning up temporary directory: {temp_dir_8bit}")
                shutil.rmtree(temp_dir_8bit)

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
            "foreground_threshold": self.foreground_threshold,
            "center_distance_threshold": self.center_distance_threshold,
            "boundary_distance_threshold": self.boundary_distance_threshold,
            "process_channels_separate": self.process_channels_separate,
            "channel_to_process": self.channel_to_process,
            "num_epochs": self.num_epochs,
            "n_objects_per_batch": self.n_objects_per_batch,
            "batch_size": self.batch_size,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework("MicroSamSegmenter", cls)
