"""micro_sam 2 automatic segmentation, through scikit-ops.

A row of its own, not models in the MicroSAM row: micro_sam 2 (SAM2, the
``hvit_*`` models) has different knobs from micro_sam 1. The op,
``skop.ops.segment.microsam2``, runs in scikit-ops' ``pytorch`` environment.
Two engines, picked by the model: the finetuned ``*_cells`` / ``*_em_*``
models use their decoder (the AIS knobs); the base SAM2 models ``hvit_t/s/b/l``
have none and segment everything object-like from a grid of point prompts
(the AMG knobs). No training yet.
"""

from dataclasses import dataclass, field

from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopSegmenter import SkopSegmenter

try:
    from skop.ops.segment.microsam2 import microsam2
except ImportError:  # scikit-ops missing, or older than this op
    microsam2 = None

#: Built-in -> axes. All read colour.
BUILTIN_MODELS = {
    "hvit_t_cells": "YXC",
    "hvit_s_cells": "YXC",
    "hvit_b_cells": "YXC",
    "hvit_l_cells": "YXC",
    "hvit_s_em_mito": "YXC",
    "hvit_s_em_nuclei": "YXC",
    # Base SAM2, no decoder: AMG.
    "hvit_t": "YXC",
    "hvit_s": "YXC",
    "hvit_b": "YXC",
    "hvit_l": "YXC",
}
BASE_MODELS = ("hvit_t", "hvit_s", "hvit_b", "hvit_l")


def _num(kind, default, low, high, step):
    return field(
        default=default,
        metadata={
            "type": kind,
            "param_type": "inference",
            "min": low,
            "max": high,
            "step": step,
            "default": default,
        },
    )


@dataclass
class MicroSam2SkopSegmenter(SkopSegmenter):
    """micro_sam 2 through scikit-ops, with the model chosen by combo."""

    #: On: each model's own tuned settings, and the knobs below are ignored.
    use_model_defaults: bool = field(
        default=True,
        metadata={"type": "bool", "param_type": "inference", "default": True},
    )
    # Starting values are hvit_t_cells' defaults.
    foreground_threshold: float = _num("float", 0.5, 0.0, 1.0, 0.05)
    density_threshold: float = _num("float", 30.0, 0.0, 500.0, 1.0)
    min_size: int = _num("int", 50, 0, 100000, 10)
    foreground_weight: float = _num("float", 0.9, 0.0, 1.0, 0.05)
    sigma: float = _num("float", 0.5, 0.0, 5.0, 0.05)
    n_iter: int = _num("int", 50, 1, 2000, 10)
    dt: float = _num("float", 1.0, 0.1, 5.0, 0.1)
    # Base models (AMG). Size limits suit ~25k px red cells; 0 max = no limit.
    pred_iou_thresh: float = _num("float", 0.8, 0.0, 1.0, 0.05)
    stability_score_thresh: float = _num("float", 0.85, 0.0, 1.0, 0.05)
    min_object_size: int = _num("int", 1000, 0, 1000000, 100)
    max_object_size: int = _num("int", 0, 0, 10000000, 1000)
    #: Off by default for base models, where tile borders cut objects.
    tiled: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    instructions = """
micro_sam 2 automatic segmentation (scikit-ops):
• hvit_t_cells - smallest and fastest, start here; s, b, l are larger
• use model defaults - each model's tuned settings; off to use the knobs
• foreground threshold - lower finds more, and fainter, objects
• density threshold - lower splits more
• min size - raise it to drop specks
base SAM2 (hvit_t/s/b/l) - segments everything, unstained cells too:
• iou / stability thresholds - lower keeps more, and worse, masks
• min / max object size - in pixels; max 0 = no limit
• tiled - off keeps whole objects; on for very large images
    """

    BUILTIN_MODELS = BUILTIN_MODELS

    def __post_init__(self):
        super().__post_init__()
        self.inference_model_name = next(iter(BUILTIN_MODELS))

    @property
    def op(self):
        return microsam2

    def get_model_axis_map(self) -> dict:
        return dict(BUILTIN_MODELS)

    def set_model(self, model_name):
        self.inference_model_name = model_name

    def get_recommended_axis(self) -> str:
        return BUILTIN_MODELS.get(self.inference_model_name, "YXC")

    def _op_params(self) -> dict:
        params = {"model": self.inference_model_name}
        if self.inference_model_name in BASE_MODELS:
            params.update(
                pred_iou_thresh=float(self.pred_iou_thresh),
                stability_score_thresh=float(self.stability_score_thresh),
                min_object_size=int(self.min_object_size),
                max_object_size=int(self.max_object_size) or None,
                tiled=bool(self.tiled),
            )
        elif not self.use_model_defaults:
            params.update(
                foreground_threshold=float(self.foreground_threshold),
                density_threshold=float(self.density_threshold),
                min_size=int(self.min_size),
                foreground_weight=float(self.foreground_weight),
                sigma=float(self.sigma),
                n_iter=int(self.n_iter),
                dt=float(self.dt),
            )
        return params

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "MicroSAM2 (scikit-ops)", cls
        )
