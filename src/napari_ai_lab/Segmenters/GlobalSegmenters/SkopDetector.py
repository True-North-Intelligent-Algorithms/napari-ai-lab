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

import os
from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopSegmenter import SkopSegmenter

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
        result = self._run(image)
        boxes = getattr(result, "boxes", result)
        self.last_napari_boxes = _to_napari(boxes)
        # One confidence per box, where the op gives them: tnia-ops' people
        # call them confidences, its bibs scores; skop's Boxes has neither.
        for attr in ("scores", "confidences"):
            scores = getattr(result, attr, None)
            if scores is not None and len(scores) == len(
                self.last_napari_boxes
            ):
                self.last_box_scores = [float(s) for s in scores]
                break
        print(f"{type(self).__name__}: {len(self.last_napari_boxes)} boxes")
        return np.zeros(image.shape[:2], dtype=np.uint16)


class PtModelPicker:
    """A YOLO ``.pt`` chosen from the project's models/ folder, or browsed.

    For a detector with a ``model_file_path`` field: the combo lists every
    ``.pt`` under the project's ``models/`` directory, and a browsed file
    wins over the combo. Shared by YOLO and tnia-ops' people detector.
    """

    def __post_init__(self):
        super().__post_init__()
        # Set by the app when the segmenter is selected.
        self.model_save_dir = ""
        self.inference_model_name = ""

    def _model_files(self) -> dict:
        """Name -> path for every .pt under model_save_dir. The name is the
        path relative to it, so two runs' best.pt stay apart."""
        if not self.model_save_dir or not os.path.isdir(self.model_save_dir):
            return {}
        found = {}
        for root, _dirs, files in sorted(os.walk(self.model_save_dir)):
            for name in sorted(files):
                if name.endswith(".pt"):
                    path = os.path.join(root, name)
                    found[os.path.relpath(path, self.model_save_dir)] = path
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
                "and pick it from the model combo, or browse to one with "
                "model file path."
            )
        return files[name]


@dataclass
class YoloSkopDetector(PtModelPicker, SkopDetector):
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
    """

    def _op_params(self) -> dict:
        return {
            "weights": self._weights(),
            "conf": self.conf,
            "iou": self.iou,
            "object_size": float(self.object_size) or None,
            "overlap": self.overlap,
        }

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
