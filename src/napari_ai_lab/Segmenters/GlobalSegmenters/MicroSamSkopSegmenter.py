"""micro_sam automatic segmentation and training, through scikit-ops.

The scikit-ops twin of ``MicroSamSegmenter``. The ops, in
``napari_ai_lab_ops.microsam_auto``, run in scikit-ops' ``pytorch``
environment, which carries micro_sam -- so this works from a napari hub
install, where micro_sam is not installed and ai-lab does not depend on it.

Models: micro_sam's built-ins by name, plus every
``<model_save_dir>/checkpoints/<name>/best.pt`` -- micro_sam's own layout,
and the one ``MicroSamSegmenter`` trains into, so models trained by either
are listed by both.
"""

import os
from dataclasses import dataclass, field

from ...mixins import TrainingBase
from .CellposeSkopSegmenterBase import CellposeSkopSegmenterBase
from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopSegmenter import SkopSegmenter, _get_runner

try:
    from napari_ai_lab_ops.microsam_auto import (
        microsam_segment,
        train_microsam,
    )
except ImportError:  # scikit-ops (opspec) missing
    microsam_segment = train_microsam = None

#: Built-in -> axes. Every one but plain vit_b has the instance decoder.
BUILTIN_MODELS = {
    "vit_b_lm": "YX",
    "vit_l_lm": "YX",
    "vit_b_em_organelles": "YX",
    "vit_b_histopathology": "YXC",
    "vit_b": "YX",
}


def _int(default, low, high, step=1, kind="inference"):
    return field(
        default=default,
        metadata={
            "type": "int",
            "param_type": kind,
            "min": low,
            "max": high,
            "step": step,
            "default": default,
        },
    )


def _float(default, low, high, step, kind="inference"):
    return field(
        default=default,
        metadata={
            "type": "float",
            "param_type": kind,
            "min": low,
            "max": high,
            "step": step,
            "default": default,
        },
    )


@dataclass
class MicroSamSkopSegmenter(SkopSegmenter, TrainingBase):
    """micro_sam through scikit-ops, with the model chosen by combo."""

    # Inference
    foreground_threshold: float = _float(0.5, 0.0, 1.0, 0.05)
    center_distance_threshold: float = _float(0.5, 0.0, 1.0, 0.05)
    boundary_distance_threshold: float = _float(0.5, 0.0, 1.0, 0.05)
    tile: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )
    tile_size: int = _int(384, 64, 4096, 32)
    halo: int = _int(64, 0, 512)

    # Training
    num_epochs: int = _int(100, 1, 5000, kind="training")
    n_objects_per_batch: int = _int(5, 1, 50, kind="training")
    train_batch_size: int = _int(1, 1, 64, kind="training")
    learning_rate: float = _float(1e-5, 0.0, 1.0, 0.000001, kind="training")
    val_size: int = _int(2, 0, 100, kind="training")
    early_stopping: int = _int(10, 0, 1000, kind="training")

    instructions = """
micro_sam automatic segmentation (scikit-ops):
• vit_b_lm - light microscopy, the one to start with
• vit_l_lm - larger and slower, sometimes better
• vit_b - plain SAM, no instance decoder: much slower
• thresholds - lower foreground finds more; distances split touching objects
• tile - for large images
• training finetunes with the instance decoder; about 8 GB GPU at vit_b
    """

    BUILTIN_MODELS = BUILTIN_MODELS

    # The patch-directory reading and the progress relay are the Cellpose
    # ones: same app layout, same updater.
    _patch_pairs = CellposeSkopSegmenterBase._patch_pairs
    _reporter = CellposeSkopSegmenterBase._reporter

    def __post_init__(self):
        super().__post_init__()
        TrainingBase.__init__(self)
        self.model_save_dir = ""
        self.training_model_name = ""
        self.inference_model_name = next(iter(BUILTIN_MODELS))
        #: Which model training starts from. Empty means vit_b_lm.
        self.initial_model_name = ""

    @property
    def op(self):
        return microsam_segment

    def get_model_axis_map(self) -> dict:
        result = dict(BUILTIN_MODELS)
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """The project's trained micro_sam models."""
        return dict.fromkeys(self._checkpoints(), "YX")

    def _checkpoints(self) -> dict:
        """Name -> best.pt, for each ``checkpoints/<name>/best.pt``."""
        if not self.model_save_dir:
            return {}
        root = os.path.join(self.model_save_dir, "checkpoints")
        if not os.path.isdir(root):
            return {}
        found = {}
        for entry in sorted(os.listdir(root)):
            best = os.path.join(root, entry, "best.pt")
            if os.path.isfile(best):
                found[entry] = best
        return found

    def set_model(self, model_name):
        self.inference_model_name = model_name

    def get_recommended_axis(self) -> str:
        return self.get_model_axis_map().get(self.inference_model_name, "YX")

    def _choice(self, name):
        """A built-in by name, or a trained checkpoint by path."""
        if name in BUILTIN_MODELS:
            return {"model": name}
        checkpoint = self._checkpoints().get(name)
        if checkpoint is None:
            raise FileNotFoundError(
                f"No micro_sam model '{name}' under {self.model_save_dir}"
            )
        # The op reads the base model from skop_microsam.json when the
        # checkpoint has one; vit_b_lm is the guess for older ones.
        return {"model": "vit_b_lm", "checkpoint": checkpoint}

    def _op_params(self) -> dict:
        params = {
            "foreground_threshold": float(self.foreground_threshold),
            "center_distance_threshold": float(self.center_distance_threshold),
            "boundary_distance_threshold": float(
                self.boundary_distance_threshold
            ),
            "tile": bool(self.tile),
            "tile_size": int(self.tile_size),
            "halo": int(self.halo),
        }
        params.update(self._choice(self.inference_model_name))
        return params

    def train(self, updater=None):
        """Train from the app's patch directory into model_save_dir."""
        if train_microsam is None:
            return {
                "success": False,
                "message": "scikit-ops is not installed -- pip install "
                "scikit-ops",
            }
        if not self.patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not self.model_save_dir:
            return {"success": False, "message": "model_save_dir is not set."}

        images, labels = self._patch_pairs()
        if not images:
            return {
                "success": False,
                "message": f"No .tif pairs found under {self.patch_path}.",
            }

        start = {"model": "vit_b_lm"}
        if self.initial_model_name:
            start = self._choice(self.initial_model_name)
            if "checkpoint" in start:
                start["initial_checkpoint"] = start.pop("checkpoint")

        name = self.training_model_name or "microsam_model"
        best = _get_runner().run(
            train_microsam,
            images=images,
            labels=labels,
            model_dir=self.model_save_dir,
            name=name,
            epochs=int(self.num_epochs),
            n_objects_per_batch=int(self.n_objects_per_batch),
            batch_size=int(self.train_batch_size),
            learning_rate=float(self.learning_rate),
            val_size=int(self.val_size),
            early_stopping=int(self.early_stopping),
            on_progress=self._reporter(updater),
            **start,
        )
        self.inference_model_name = name
        return {
            "success": True,
            "message": f"Training complete. Model saved to: {best}",
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "MicroSAM (scikit-ops)", cls
        )
