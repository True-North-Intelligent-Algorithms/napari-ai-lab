"""scikit-ops detectors: ops that return boxes instead of a label image.

Every op in ``skop.ops.detect`` returns the same ``Boxes``, so one base does
the work and each subclass only names its op and the parameters it exposes.
One subclass per op rather than one per model: each op is pinned to its own
environment, and a model -- a trained ``.pt`` -- is a parameter of the op.

The boxes reach the viewer the way MicroSamYoloSegmenter's do, through
``last_napari_boxes``, which the segment widget turns into a
"Box predictions (<name>)" layer. ``segment`` still has to return a mask, so
it returns an empty one; a profile without labels drops it.
"""

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np

from ...mixins import TrainingBase
from ...utilities.shared_models import shared_models_dir
from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopSegmenter import SkopSegmenter, _get_runner

# One at a time: a scikit-ops without one of them should still offer the others.
try:
    from skop.ops.detect import fastsam
    from skop.ops.detect.fastsam import PretrainedModel as FastSamModel
except ImportError:
    fastsam = FastSamModel = None
try:
    from skop.ops.detect import yolo
except ImportError:
    yolo = None
try:
    from skop.ops.detect import object_aware_yolo
except ImportError:
    object_aware_yolo = None
try:
    # Apart from the detect op: a scikit-ops without the training op should
    # lose the Train button, not the detector.
    from skop.ops.train import train_yolo
    from skop.ops.train.yolo import Cache as YoloCache
except ImportError:
    train_yolo = YoloCache = None

#: Stock models training can start from; ultralytics downloads them. Bigger
#: is slower and, with enough boxes, better: n, s, m.
STOCK_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]


def _training_field(default, kind, **meta):
    return field(
        default=default,
        metadata={
            "type": kind,
            "param_type": "training",
            "default": default,
            **meta,
        },
    )


def _float_field(default, lo=0.0, hi=1.0, step=0.05):
    return field(
        default=default,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": lo,
            "max": hi,
            "step": step,
            "default": default,
        },
    )


def _int_field(default, lo, hi, step):
    return field(
        default=default,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": lo,
            "max": hi,
            "step": step,
            "default": default,
        },
    )


def _to_napari(boxes) -> list[np.ndarray]:
    """(N, 4) [min_y, min_x, max_y, max_x] -> napari rectangles, 4 vertices each."""
    rects = []
    for y0, x0, y1, x1 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        rects.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]))
    return rects


@dataclass
class SkopDetector(SkopSegmenter):
    """Base for segmenters backed by a skop detect op. Not registered."""

    def segment(self, image, **kwargs):
        """Run the op, keep its boxes for the viewer, return an empty mask."""
        if not self.are_dependencies_available():
            raise ImportError(
                "This detector requires scikit-ops, which is not installed. "
                "Install it with: pip install scikit-ops"
            )
        self.last_napari_boxes = None
        self.last_box_scores = None
        self.last_box_classes = None
        self.last_box_class_names = None
        result = self._run(image)
        boxes = getattr(result, "boxes", result)
        self.last_napari_boxes = _to_napari(boxes)
        n = len(self.last_napari_boxes)
        # One confidence per box, where the op gives them: tnia-ops' people
        # call them confidences, its bibs scores; skop's Boxes has neither.
        for attr in ("scores", "confidences"):
            scores = getattr(result, attr, None)
            if scores is not None and len(scores) == n:
                self.last_box_scores = [float(s) for s in scores]
                break
        # And a class per box, where it gives them (skop's Detections). They
        # mean something only when the classes have names.
        classes = getattr(result, "classes", None)
        if classes is not None and len(classes) == n:
            self.last_box_classes = [int(c) for c in classes]
            self.last_box_class_names = self._class_names() or None
        print(f"{type(self).__name__}: {n} boxes")
        return np.zeros(image.shape[:2], dtype=np.uint16)

    def _class_names(self) -> dict:
        """Class id -> name for the model just run; {} when unknown."""
        return {}


def trained_class_names(weights) -> dict:
    """The class names a model trained by train_yolo was trained with.

    train_yolo keeps its working files in ``<name>_training/`` beside
    ``<name>.pt``, and the dataset's data.yaml, which names the classes, is
    one of them. {} for a stock model, or one copied in without them.
    """
    from pathlib import Path

    from ...utilities.box_annotations import read_names

    path = Path(weights)
    return read_names(
        path.with_name(f"{path.stem}_training") / "dataset" / "data.yaml"
    )


class PtModelPicker:
    """A YOLO ``.pt`` chosen from the project's or the shared models folder, or browsed.

    For a detector with a ``model_file_path`` field: the combo lists every
    ``.pt`` under the project's ``models/`` directory and in the per-user
    shared one (``utilities.shared_models``), and a browsed file
    wins over the combo. Shared by YOLO and tnia-ops' people detector.
    """

    def __post_init__(self):
        super().__post_init__()
        # Set by the app when the segmenter is selected.
        self.model_save_dir = ""
        self.inference_model_name = ""

    def _model_files(self) -> dict:
        """Name -> path for every .pt under model_save_dir, then every one in
        the per-user shared models folder. A project model's name is its
        path relative to model_save_dir, so two runs' best.pt stay apart; a
        shared one is marked "(shared)"."""
        found = {}
        if self.model_save_dir and os.path.isdir(self.model_save_dir):
            # Walked top-down, unsorted, so the folders can be pruned as it
            # goes: archived models and training's working folders
            # (<name>_training/) are not models to choose.
            for root, dirs, files in os.walk(self.model_save_dir):
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if d != "archive" and not d.endswith("_training")
                )
                for name in sorted(files):
                    if name.endswith(".pt"):
                        path = os.path.join(root, name)
                        found[os.path.relpath(path, self.model_save_dir)] = (
                            path
                        )
        shared = shared_models_dir()
        if shared.is_dir():
            for path in sorted(shared.glob("*.pt")):
                found[f"{path.name} (shared)"] = str(path)
        return found

    def get_model_axis_map(self) -> dict:
        """Every .pt found, for the model combo. YOLO takes grey or colour."""
        return dict.fromkeys(self._model_files(), "YXC")

    def set_model(self, model_name):
        self.inference_model_name = model_name

    def get_recommended_axis(self) -> str:
        return "YXC"

    def _weights(self) -> str:
        """The browsed file if one is set, else the combo's pick."""
        if self.model_file_path:
            if not os.path.isfile(self.model_file_path):
                raise FileNotFoundError(
                    f"YOLO model file not found: {self.model_file_path!r}"
                )
            return self.model_file_path
        files = self._model_files()
        name = self.inference_model_name or next(iter(files), "")
        if name not in files:
            raise FileNotFoundError(
                f"No YOLO model chosen. Put a .pt file in {self.model_save_dir!r} "
                f"or {str(shared_models_dir())!r} "
                "and pick it from the model combo, or browse to one with "
                "model file path."
            )
        return files[name]


@dataclass
class YoloSkopDetector(PtModelPicker, SkopDetector, TrainingBase):
    """Any trained YOLO model, on the whole image or in tiles: skop's ``yolo``.

    The main YOLO detector. FastSAM and object-aware YOLO are the special
    ones, each with its own fixed model. The model is picked from a combo of every ``.pt`` under the project's
    ``models/`` directory, the way the Cellpose segmenters list theirs -- so
    a stock download and a training run's ``weights/best.pt`` both show up
    without browsing. A model kept anywhere else is picked with the model
    file browser, the way the MONAI UNet takes one, and wins over the combo.
    """

    model_file_path: str = field(
        default="",
        metadata={
            "type": "file",
            "param_type": "inference",
            "file_type": "file",
            "default": "",
        },
    )
    conf: float = _float_field(0.25)
    iou: float = _float_field(0.5)
    object_size: int = _int_field(0, 0, 10_000_000, 100)
    overlap: float = _float_field(0.15, 0.0, 0.95)

    # Training, on a crop set the Augment tab's "YOLO crops" wrote.
    num_epochs: int = _training_field(100, "int", min=1, max=5000, step=10)
    #: 0 trains at the size the crops were cut at, which is the usual thing.
    train_imgsz: int = _training_field(0, "int", min=0, max=4096, step=32)
    #: -1 lets ultralytics choose from the GPU's memory.
    train_batch_size: int = _training_field(16, "int", min=-1, max=256, step=1)
    #: What "start from" means when no model of the project's is chosen.
    stock_model: str = _training_field(
        "yolo11s.pt", "str", choices=STOCK_MODELS
    )
    patience: int = _training_field(50, "int", min=0, max=5000, step=10)
    #: 0: numbers and letters must not be mirrored. 0.5 for symmetric things.
    fliplr: float = _training_field(0.0, "float", min=0.0, max=1.0, step=0.1)
    #: Ultralytics' own augmentation on top of the crops' (mosaic, zoom...).
    augment: bool = _training_field(True, "bool")
    cache: str = _training_field("ram", "str", choices=["ram", "disk", "off"])

    #: Shown in the "Start from" combo for no project model: training starts
    #: from the stock model below, not from nothing.
    FROM_SCRATCH_LABEL = "(stock model)"

    #: What the panel shows; "More..." shows the instructions below.
    summary = """
YOLO (scikit-ops): finds objects with a .pt model from models/.
Small objects in big photos: set object size. Lower conf finds more.
Train: make YOLO crops first (Augment tab), pick them as Patches.
    """

    instructions = """
YOLO (scikit-ops):
• model - any YOLO .pt in the project's models/ folder, subfolders included
• model file path - a .pt from anywhere else; used instead of the combo
  when set
• a stock COCO model finds COCO things (elephant, zebra, ...); cells and
  coins need a trained one
• conf - lower finds more
• iou - two boxes overlapping more than this count as one
• each image or tile is resized to the size the model was trained at
• object size - a typical object's area in pixels (a 60 x 80 bib: 4800).
  0 runs the whole image at once; set it when objects are small in a big
  image, and the image is cut into tiles that keep them big enough
• overlap - how much neighbouring tiles share, so an object on a seam is
  whole in one of them
Training: make crops first (Augment tab, YOLO crops), then pick them as
"Patches to Process". The model is saved as models/<name>.pt.
    """

    def __post_init__(self):
        super().__post_init__()
        TrainingBase.__init__(self)
        self.training_model_name = ""
        # Which model training starts from; "" is the stock model.
        self.initial_model_name = ""

    def train(self, updater=None):
        """Train on the crop set the app selected (``patch_path``).

        The crop set's pairs.json lists the images and labels, training
        first and validation last, and the size they were cut at; this hands
        them to scikit-ops' ``train_yolo``, which writes
        ``<model_save_dir>/<name>.pt`` and its history beside it.
        """
        from ...utilities.box_crops import read_pairs

        if train_yolo is None:
            return {
                "success": False,
                "message": "scikit-ops has no YOLO training op -- upgrade it.",
            }
        if not self.model_save_dir:
            return {"success": False, "message": "model_save_dir is not set."}
        if not self.patch_path:
            return {
                "success": False,
                "message": (
                    "Please augment first: in the Augment tab, choose YOLO "
                    "crops and press Perform Augmentations. Then pick the "
                    "crops here, in Patches to Process."
                ),
            }
        if not os.path.isfile(os.path.join(self.patch_path, "pairs.json")):
            return {
                "success": False,
                "message": (
                    f"'{os.path.basename(self.patch_path)}' is not YOLO "
                    "crops. In Patches to Process, pick the crops YOLO crops "
                    "made (yolo_<box layer>), or make them in the Augment "
                    "tab: choose YOLO crops and press Perform Augmentations."
                ),
            }
        pairs = read_pairs(self.patch_path)
        name = self.training_model_name or "yolo_model"

        def on_progress(event):
            if updater is None:
                return
            # See StardistSkopSegmenter.train: an event can arrive after the
            # progress bar behind updater is gone.
            with contextlib.suppress(RuntimeError):
                updater(
                    event.current or 0,
                    event.maximum or self.num_epochs,
                    event.message or "",
                )

        from qtpy.QtCore import QCoreApplication
        from superqt.utils import ensure_main_thread

        report = (
            ensure_main_thread(on_progress)
            if QCoreApplication.instance() is not None
            else on_progress
        )

        try:
            model_path = self._run_training(pairs, name, report)
        except Exception as e:  # noqa: BLE001 -- appose's TaskException
            if "crashed" not in str(e):
                raise
            return {"success": False, "message": self._crash_message(name, e)}

        # Selectable without a restart: the model combo lists models/*.pt.
        self.inference_model_name = os.path.relpath(
            model_path, self.model_save_dir
        )
        return {
            "success": True,
            "message": f"Training complete. Model saved to: {model_path}",
        }

    def _crash_message(self, name, error) -> str:
        """What to say when the training process died: how far it got, and
        the last epoch's weights to carry on from."""
        workspace = os.path.join(self.model_save_dir, f"{name}_training")
        last = os.path.join(workspace, "run", "weights", "last.pt")
        epochs = 0
        results = os.path.join(workspace, "run", "results.csv")
        if os.path.exists(results):
            with open(results) as f:
                epochs = max(0, sum(1 for _ in f) - 1)
        first = str(error).strip().splitlines()[0]
        message = (
            f"The training process crashed after {epochs} of "
            f"{self.num_epochs} epochs ({first})"
        )
        if os.path.exists(last):
            message += (
                f". The last weights it saved are in {last}: copy that into "
                f"models/ under a new name and start from it to carry on"
            )
        return (
            message + ". Its log: " + os.path.join(workspace, "training.log")
        )

    def _run_training(self, pairs, name, report):
        return _get_runner().run(
            train_yolo,
            images=pairs["images"],
            labels=pairs["labels"],
            model_dir=self.model_save_dir,
            name=name,
            class_names=pairs.get("class_names") or None,
            initial_model=self._initial_weights(),
            epochs=self.num_epochs,
            imgsz=self.train_imgsz or int(pairs.get("imgsz", 640)),
            batch_size=self.train_batch_size,
            patience=self.patience,
            fliplr=self.fliplr,
            augment=self.augment,
            cache=YoloCache[self.cache],
            val_size=int(pairs["val_size"]),
            dataset_id=f"{os.path.basename(self.patch_path)} {pairs.get('made', '')}",
            on_progress=report,
        )

    def _initial_weights(self) -> str:
        """A project or shared model's path, or the stock model's name."""
        if self.initial_model_name:
            return self._model_files().get(
                self.initial_model_name, self.initial_model_name
            )
        return self.stock_model

    def _op_params(self) -> dict:
        return {
            "weights": self._weights(),
            "conf": self.conf,
            "iou": self.iou,
            "object_size": float(self.object_size) or None,
            "overlap": self.overlap,
        }

    def _class_names(self) -> dict:
        return trained_class_names(self._weights())

    @property
    def op(self):
        return yolo

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework("YOLO (scikit-ops)", cls)


@dataclass
class FastSamSkopDetector(SkopDetector):
    """FastSAM's detector: every object, one class."""

    model: str = field(
        default="small",
        metadata={
            "type": "str",
            "param_type": "inference",
            "default": "small",
            "choices": ["small", "large"],
        },
    )
    conf: float = _float_field(0.4)
    iou: float = _float_field(0.9)
    max_det: int = _int_field(300, 1, 5000, 50)
    imgsz: int = _int_field(1024, 32, 4096, 32)

    instructions = """
FastSAM (scikit-ops):
• finds every object, without saying what it is
• conf - lower finds more
• iou - two boxes overlapping more than this count as one
• imgsz - the image is resized to this
    """

    @property
    def op(self):
        return fastsam

    def _op_params(self) -> dict:
        return {
            "model": FastSamModel[self.model],
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "imgsz": self.imgsz,
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "FastSAM (scikit-ops)", cls
        )


@dataclass
class ObjectAwareYoloSkopDetector(SkopDetector):
    """segment-everything's object-aware YOLO: every object, one class."""

    conf: float = _float_field(0.4)
    iou: float = _float_field(0.9)
    max_det: int = _int_field(400, 1, 5000, 50)
    imgsz: int = _int_field(1024, 32, 4096, 32)

    instructions = """
Object-aware YOLO (scikit-ops):
• finds every object, without saying what it is
• runs in its own environment, built on first use
• conf - lower finds more
• imgsz - the image is resized to this
    """

    @property
    def op(self):
        return object_aware_yolo

    def _op_params(self) -> dict:
        return {
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            "imgsz": self.imgsz,
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "Object-aware YOLO (scikit-ops)", cls
        )
