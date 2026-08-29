"""StarDist as a scikit-ops segmenter, with a model combo.

``SkopSegmenter.register_op`` builds one segmenter per op, which put two
StarDist rows in the list -- one per pretrained model. That is not how the
other segmenters work: ``StardistSegmenter`` is a single entry with a combo
listing the models, builtin and user-trained alike. This class follows that,
so there is one StarDist (scikit-ops) row and the choice of model happens
inside it.

Which op runs is a private detail of ``segment()``. The user picks a *model*;
several models map onto one op, and every user-trained model will map onto
``stardist2d_custom``

The model-map methods are duplicated from ``StardistSegmenter`` rather than
shared, deliberately
"""

import contextlib
import json
import os
from dataclasses import dataclass, field

from superqt.utils import ensure_main_thread

from ...mixins import TrainingBase
from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopSegmenter import SkopSegmenter, _get_runner

try:
    from skop.ops.segment.stardist2d import (
        stardist2d_custom,
        stardist2d_fluo,
        stardist2d_he,
    )

    _is_stardist_op_available = True
except ImportError:  # scikit-ops absent, or too old to carry these ops
    stardist2d_fluo = stardist2d_he = stardist2d_custom = None
    _is_stardist_op_available = False

try:
    # Separate from the inference ops on purpose: a scikit-ops without the
    # training op should lose the Train button, not the whole segmenter.
    from skop.ops.train import train_stardist2d
except ImportError:
    train_stardist2d = None


#: Builtin model -> the op that runs it. The 2D ops only; 3D is a separate op
#: in scikit-ops and would need its own entry here.
BUILTIN_OPS = {
    "2D_versatile_fluo": stardist2d_fluo,
    "2D_versatile_he": stardist2d_he,
}

#: Builtin model -> the axes it expects. Same values as StardistSegmenter's
#: map, minus 3D_demo, which has no op here yet.
BUILTIN_MODEL_MAP = {
    "2D_versatile_fluo": "YX",  # Grayscale fluorescence
    "2D_versatile_he": "YXC",  # RGB H&E staining
}


@dataclass
class StardistSkopSegmenter(SkopSegmenter, TrainingBase):
    """StarDist 2D through scikit-ops, with the model chosen by combo."""

    # Inference parameters. All three ops take the same three, so the form
    # does not change as the model combo does. Declared here rather than
    # generated from the op signature: which parameters a user should see is
    # ai-lab's call, and one widget generator is enough.
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
    normalize: bool = field(
        default=True,
        metadata={"type": "bool", "param_type": "inference", "default": True},
    )

    # Training hyper-parameters.
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
    steps_per_epoch: int = field(
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
    train_patch_size_xy: int = field(
        default=128,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 32,
            "max": 2048,
            "step": 32,
            "default": 128,
        },
    )
    train_batch_size: int = field(
        default=4,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 64,
            "step": 1,
            "default": 4,
        },
    )
    unet_n_depth: int = field(
        default=3,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 6,
            "step": 1,
            "default": 3,
        },
    )
    grid_size_xy: int = field(
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

    #: Baked into a trained model, so continuing from one inherits them and
    #: the op ignores whatever the form says. The training tab greys these
    #: out while a starting model is selected.
    INHERITED_WHEN_CONTINUING = (
        "grid_size_xy",
        "unet_n_depth",
        "train_patch_size_xy",
        "train_batch_size",
    )

    instructions = """
StarDist 2D (scikit-ops):
• Runs in scikit-ops' own environment, not this one
• The first run builds that environment, which takes minutes and gigabytes
• Model: choose a pretrained model, or one trained into this project
    """

    def __post_init__(self):
        super().__post_init__()
        TrainingBase.__init__(self)
        # Set by the app before segment() and train().
        self.model_save_dir = ""
        self.training_model_name = ""
        self.inference_model_name = "2D_versatile_fluo"
        # Which model training starts from. Empty means random weights.
        # Deliberately not the same attribute as inference_model_name: the
        # two combos ask different questions of the same list.
        self.initial_model_name = ""

    # ------------------------------------------------------------------
    # Model list. NDOperationWidget adds a "Model:" combo to any segmenter
    # that has get_model_axis_map, and calls set_model when it changes, so
    # defining these three is all the combo needs.
    # ------------------------------------------------------------------

    def get_model_axis_map(self) -> dict:
        """Every selectable model, mapped to the axes it expects."""
        result = BUILTIN_MODEL_MAP.copy()
        result.update(self.build_pretrained_model_map())
        return result

    def build_pretrained_model_map(self) -> dict:
        """Scan ``model_save_dir`` for models trained into this project.

        A subdirectory is a model when it holds a ``config.json`` with
        ``axes``. The trailing ``C`` is dropped when the model takes a single
        channel, so the axis combo offers what the model actually wants.
        """
        if not self.model_save_dir or not os.path.isdir(self.model_save_dir):
            return {}

        pretrained = {}
        for entry in os.listdir(self.model_save_dir):
            config_path = os.path.join(
                self.model_save_dir, entry, "config.json"
            )
            if not os.path.isfile(config_path):
                continue
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                axes = cfg.get("axes", "YX")
                if cfg.get("n_channel_in", 1) == 1 and axes.endswith("C"):
                    axes = axes[:-1]
                pretrained[entry] = axes
            except (json.JSONDecodeError, OSError):
                continue
        return pretrained

    def set_model(self, model_name):
        """Select a model. Nothing is loaded here -- the worker holds it."""
        self.inference_model_name = model_name

    def get_recommended_axis(self) -> str:
        return self.get_model_axis_map().get(self.inference_model_name, "YX")

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    @property
    def op(self):
        """The op for the selected model.

        A builtin runs its own op; everything else is a model on disk and
        runs ``stardist2d_custom``. All three share the 'stardist-tf'
        environment and the same parameters, so the availability banner and
        the generated form do not change as the combo does.
        """
        return BUILTIN_OPS.get(self.inference_model_name, stardist2d_custom)

    def _op_params(self) -> dict:
        """The three inference fields, plus the model directory for a custom
        model. ``model_dir`` is not a form field: the model combo chose it,
        and a directory box beside the combo would be a second, disagreeing
        model selector.
        """
        params = {
            "prob_thresh": self.prob_thresh,
            "nms_thresh": self.nms_thresh,
            "normalize": self.normalize,
        }
        if self.inference_model_name not in BUILTIN_OPS:
            params["model_dir"] = self._model_dir()
        return params

    def _model_dir(self) -> str:
        """Where the selected user-trained model lives.

        ai-lab settled how a project stores models -- ``<project>/models/
        <name>``, a directory for StarDist -- so resolving a name to a path
        is one join, and scikit-ops never learns the layout.
        """
        return os.path.join(self.model_save_dir, self.inference_model_name)

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "StarDist2D (scikit-ops)", cls
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, updater=None):
        """Train a model from the patch directory the app selected.

        The layout knowledge stops here: this reads ``info.json`` for the
        axes and pairs ``input0`` with ``ground_truth0``, then hands the op
        two lists of paths. scikit-ops never learns what a patch directory
        looks like -- design 0011.
        """
        if train_stardist2d is None:
            return {
                "success": False,
                "message": "scikit-ops has no training ops -- upgrade it.",
            }
        if not self.patch_path:
            return {"success": False, "message": "patch_path is not set."}
        if not self.model_save_dir:
            return {"success": False, "message": "model_save_dir is not set."}

        name = self.training_model_name or "stardist_model"

        with open(os.path.join(self.patch_path, "info.json")) as f:
            info = json.load(f)
        axes = info["axes"]

        images, labels = self._patch_pairs()
        if not images:
            return {
                "success": False,
                "message": f"No .tif pairs found under {self.patch_path}.",
            }

        # Progress arrives on Appose's pipe-reading thread, which Qt knows
        # nothing about, so calling `updater` from it reaches a widget from a
        # foreign thread -- the reason this reported nothing at all. The
        # in-process segmenters do not need this: they call `updater` from the
        # QThread that TrainingThread created, which Qt can queue from.
        # skop_napari/_run.py does the same bounce for the same reason.
        @ensure_main_thread
        def on_progress(event):
            if updater is None:
                return
            # Queueing to the main thread means an event can arrive after
            # training finished and the worker behind `updater` was deleted,
            # leaving a Qt object whose C++ half is gone. Nothing is lost by
            # dropping it -- there is no longer a progress bar to move.
            with contextlib.suppress(RuntimeError):
                updater(
                    event.current or 0,
                    event.maximum or self.num_epochs,
                    event.message or "",
                )

        model_path = _get_runner().run(
            train_stardist2d,
            images=images,
            labels=labels,
            model_dir=self.model_save_dir,
            name=name,
            image_axes=axes,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            train_patch_size=self.train_patch_size_xy,
            train_batch_size=self.train_batch_size,
            unet_n_depth=self.unet_n_depth,
            grid_size_xy=self.grid_size_xy,
            initial_model=self._initial_model(),
            dataset_id=info.get("dataset_id", ""),
            val_size=self.val_size,
            on_progress=on_progress,
        )

        # Make it selectable without a restart: the combo rebuilds from
        # get_model_axis_map, which scans model_save_dir.
        self.inference_model_name = name
        return {
            "success": True,
            "message": f"Training complete. Model saved to: {model_path}",
        }

    def _initial_model(self) -> str:
        """What training starts from: a directory, a builtin name, or "".

        A project model is resolved to its directory here, where the layout
        is known. A builtin is passed on by name -- resolving it means asking
        StarDist where it cached the download, and StarDist lives in the op's
        environment, not this one.
        """
        if not self.initial_model_name:
            return ""
        if self.initial_model_name in BUILTIN_MODEL_MAP:
            return self.initial_model_name
        return os.path.join(self.model_save_dir, self.initial_model_name)

    def _patch_pairs(self):
        """``input0``/``ground_truth0`` resolved to two lists of paths.

        Labels are derived from the inputs by name rather than globbed
        separately: two independent sorts agree until one directory holds an
        extra file, and the symptom then is silently mispaired training data
        rather than an error.
        """
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
