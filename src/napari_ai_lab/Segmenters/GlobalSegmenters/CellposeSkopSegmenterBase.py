"""Shared by the Cellpose 3 and Cellpose 4 scikit-ops segmenters.

Two rows, not one: the versions need different environments and cannot load
each other's models. Here: the fields, the model scan, train() and progress.
A subclass names its ops, built-ins and extra parameters.
"""

import contextlib
import json
import os
from dataclasses import dataclass, field

from superqt.utils import ensure_main_thread

from ...mixins import TrainingBase
from .SkopSegmenter import SkopSegmenter, _get_runner

try:
    # Reads a checkpoint's keys to say which Cellpose can load it.
    from skop.models import cellpose_flavor
except ImportError:
    cellpose_flavor = None


@dataclass
class CellposeSkopSegmenterBase(SkopSegmenter, TrainingBase):
    """Base for the CPSAM and Cellpose 3 scikit-ops segmenters."""

    # Inference. Both ops take these with the same meanings.
    #: 0 means "estimate it", which is harder to notice going wrong.
    diameter: float = field(
        default=30.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": 0.0,
            "max": 1000.0,
            "step": 1.0,
            "default": 30.0,
        },
    )
    flow_threshold: float = field(
        default=0.4,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": 0.0,
            "max": 3.0,
            "step": 0.05,
            "default": 0.4,
        },
    )
    cellprob_threshold: float = field(
        default=0.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": -6.0,
            "max": 6.0,
            "step": 0.1,
            "default": 0.0,
        },
    )
    #: 0 lets Cellpose scale it to the diameter.
    niter: int = field(
        default=200,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 5000,
            "step": 50,
            "default": 200,
        },
    )
    min_size: int = field(
        default=15,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 10000,
            "step": 1,
            "default": 15,
        },
    )
    normalize: bool = field(
        default=True,
        metadata={"type": "bool", "param_type": "inference", "default": True},
    )
    use_gpu: bool = field(
        default=True,
        metadata={"type": "bool", "param_type": "inference", "default": True},
    )

    # Training.
    num_epochs: int = field(
        default=100,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 5000,
            "step": 1,
            "default": 100,
        },
    )
    #: Step sets the widget's decimals, so 1e-4 to leave room an order of
    #: magnitude below the 0.005 Cellpose 3 trains at.
    learning_rate: float = field(
        default=0.005,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
            "default": 0.005,
        },
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.00001,
            "default": 1e-5,
        },
    )
    train_batch_size: int = field(
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
    train_patch_size_xy: int = field(
        default=224,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 64,
            "max": 1024,
            "step": 32,
            "default": 224,
        },
    )
    min_train_masks: int = field(
        default=5,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 0,
            "max": 100,
            "step": 1,
            "default": 1,
        },
    )
    #: 0 means every patch. Lower it when an epoch over all of them is too
    #: long to watch.
    nimg_per_epoch: int = field(
        default=0,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 0,
            "max": 100000,
            "step": 10,
            "default": 0,
        },
    )
    #: Resizes patches to the diameter the network expects.
    rescale: bool = field(
        default=True,
        metadata={"type": "bool", "param_type": "training", "default": True},
    )
    val_size: int = field(
        default=2,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 2,
        },
    )

    # Class attributes, not fields: annotating them would make them
    # per-instance and hide the subclass's value.
    BUILTIN_MODELS = {}  # name -> axes
    FLAVOR = ""  # which cellpose_flavor this segmenter lists
    TRAIN_OP = None

    def __post_init__(self):
        super().__post_init__()
        TrainingBase.__init__(self)
        # Set by the app before segment() and train().
        self.model_save_dir = ""
        self.training_model_name = ""
        self.inference_model_name = next(iter(self.BUILTIN_MODELS))
        # Which model training starts from. Empty means the default built-in.
        self.initial_model_name = ""

    def get_model_axis_map(self) -> dict:
        """Every selectable model, mapped to the axes it expects."""
        result = dict(self.BUILTIN_MODELS)
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """The project's own models of this flavor. Anything unrecognized --
        a history csv, a StarDist directory -- is skipped."""
        axes = next(iter(self.BUILTIN_MODELS.values()))
        return dict.fromkeys(self._model_files(), axes)

    def _model_files(self) -> dict:
        """Name -> path. Two directories, because Cellpose writes into a
        ``models/`` subdirectory of wherever it trains."""
        if cellpose_flavor is None or not self.model_save_dir:
            return {}

        found = {}
        for directory in (
            self.model_save_dir,
            os.path.join(self.model_save_dir, "models"),
        ):
            if not os.path.isdir(directory):
                continue
            for entry in sorted(os.listdir(directory)):
                path = os.path.join(directory, entry)
                if not os.path.isfile(path):
                    continue
                try:
                    if cellpose_flavor(path).value != self.FLAVOR:
                        continue
                except (ValueError, OSError):
                    continue
                found[entry] = path
        return found

    def set_model(self, model_name):
        """Select a model. Nothing is loaded here -- the worker holds it."""
        self.inference_model_name = model_name

    def get_recommended_axis(self) -> str:
        return self.get_model_axis_map().get(self.inference_model_name, "YXC")

    def _model_path(self, name: str) -> str:
        """Where a user-trained model lives, from the scan that offered it."""
        return self._model_files().get(
            name, os.path.join(self.model_save_dir, name)
        )

    def _op_params(self) -> dict:
        """The shared inference fields, plus whichever model was chosen."""
        params = {
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "niter": self.niter,
            "min_size": self.min_size,
            "normalize": self.normalize,
            "use_gpu": self.use_gpu,
        }
        params.update(self._model_params())
        return params

    def _model_params(self) -> dict:
        """How the selected model reaches the op. Subclass decides."""
        raise NotImplementedError

    def train(self, updater=None):
        """Train from the app's patch directory. Layout knowledge stops here:
        the op gets two lists of paths -- design 0011."""
        if self.TRAIN_OP is None:
            return {
                "success": False,
                "message": "scikit-ops has no Cellpose training op -- "
                "upgrade it.",
            }
        if not self.patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not self.model_save_dir:
            return {"success": False, "message": "model_save_dir is not set."}

        name = self.training_model_name or "cellpose_model"

        with open(os.path.join(self.patch_path, "info.json")) as f:
            info = json.load(f)

        images, labels = self._patch_pairs()
        if not images:
            return {
                "success": False,
                "message": f"No .tif pairs found under {self.patch_path}.",
            }

        model_path = _get_runner().run(
            self.TRAIN_OP,
            images=images,
            labels=labels,
            model_dir=self.model_save_dir,
            name=name,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.train_batch_size,
            patch_size=self.train_patch_size_xy,
            min_train_masks=self.min_train_masks,
            normalize=self.normalize,
            rescale=self.rescale,
            nimg_per_epoch=self.nimg_per_epoch,
            val_size=self.val_size,
            dataset_id=info.get("dataset_id", ""),
            on_progress=self._reporter(updater),
            **self._train_params(),
        )

        # Selectable without a restart: the combo rescans model_save_dir.
        self.inference_model_name = os.path.basename(model_path)
        return {
            "success": True,
            "message": f"Training complete. Model saved to: {model_path}",
        }

    def _train_params(self) -> dict:
        """What training starts from, and any channel arguments. Subclass."""
        raise NotImplementedError

    def _reporter(self, updater):
        """Bounce progress to the GUI thread. It arrives on Appose's reader
        thread, which Qt cannot be touched from; a notebook has no Qt app."""
        if updater is None:
            return None

        def on_progress(event):
            # An event can outlive the widget behind `updater`; dropping it
            # loses nothing.
            with contextlib.suppress(RuntimeError):
                updater(
                    event.current or 0,
                    event.maximum or self.num_epochs,
                    event.message or "",
                )

        from qtpy.QtCore import QCoreApplication

        if QCoreApplication.instance() is not None:
            return ensure_main_thread(on_progress)
        return on_progress

    def _patch_pairs(self):
        """``input0``/``ground_truth0`` as two lists. Labels are matched by
        name, not by sorting both -- a stray file silently mispairs them."""
        input_dir = os.path.join(self.patch_path, "input0")
        truth_dir = os.path.join(self.patch_path, "ground_truth0")

        images, labels = [], []
        for entry in sorted(os.listdir(input_dir)):
            if not entry.endswith(".tif"):
                continue
            truth = os.path.join(truth_dir, entry)
            if not os.path.isfile(truth):
                raise FileNotFoundError(
                    f"{entry} has no ground truth: {truth} does not exist."
                )
            images.append(os.path.join(input_dir, entry))
            labels.append(truth)
        return images, labels
