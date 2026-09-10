"""Cellpose 4 as a scikit-ops segmenter, with a model combo.

Four built-ins over two backbones, SAM and DINOv3. Cellpose treats them
identically, so which suits your images is a question for your images.
Three are ViT-L; cpdino_vitb is the small one.

These models read colour themselves, so the only channel choice is whether
to collapse it.
"""

from dataclasses import dataclass, field

from .CellposeSkopSegmenterBase import CellposeSkopSegmenterBase
from .GlobalSegmenterBase import GlobalSegmenterBase

try:
    from skop.ops.segment.cellpose4 import PretrainedModel, cellpose4
except ImportError:
    cellpose4 = PretrainedModel = None

try:
    # Separate: a scikit-ops without the training op loses the Train button,
    # not the segmenter.
    from skop.ops.train import train_cellpose4
except ImportError:
    train_cellpose4 = None


#: Built-in -> axes. All read colour, so hand them YXC when there is any.
BUILTIN_MODELS = {
    "cpsam_v2": "YXC",
    "cpsam": "YXC",
    "cpdino": "YXC",
    "cpdino_vitb": "YXC",
}


@dataclass
class Cellpose4SkopSegmenter(CellposeSkopSegmenterBase):
    """Cellpose 4 through scikit-ops, with the model chosen by combo."""

    #: On only when one colour is noise; these models use all three.
    collapse_channels: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    # Cellpose 4's training defaults, not the base's Cellpose 3 ones: a
    # ViT-L at batch 8 runs a laptop GPU out of memory.
    #: Step sets the widget's decimals, so 1e-6 to leave room an order of
    #: magnitude below the 1e-5 cellpose finetunes at.
    learning_rate: float = field(
        default=1e-5,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.000001,
            "default": 1e-5,
        },
    )
    weight_decay: float = field(
        default=0.1,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.1,
        },
    )
    train_batch_size: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 64,
            "step": 1,
            "default": 1,
        },
    )
    train_patch_size_xy: int = field(
        default=256,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 64,
            "max": 1024,
            "step": 32,
            "default": 256,
        },
    )
    #: Off: these models were trained across scales.
    rescale: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "training", "default": False},
    )

    BUILTIN_MODELS = BUILTIN_MODELS
    FLAVOR = "cellpose4"
    # staticmethod, or `self.TRAIN_OP` is a bound method and skop cannot read
    # its spec.
    TRAIN_OP = staticmethod(train_cellpose4)

    instructions = """
Cellpose 4:
• cell probability - -6 to 6, lower finds more
• flow threshold - 0 to 3, higher finds more, 0 turns it off (finds more)
• diameter - low finds small, high finds big. Not as important in cp4.
• iters - higher runs more flow iterations, finds larger
• model - cpsam and cpdino are different backbones, try both
• training needs about 12 GB GPU, except cpdino_vitb, about 4 GB
    """

    @property
    def op(self):
        return cellpose4

    def _model_params(self) -> dict:
        """A built-in by name, or a checkpoint by path -- never both. A
        finetuned file carries its own backbone."""
        params = {"collapse_channels": self.collapse_channels}
        if self.inference_model_name in BUILTIN_MODELS:
            params["model"] = PretrainedModel[self.inference_model_name]
        else:
            params["pretrained_model"] = self._model_path(
                self.inference_model_name
            )
        return params

    def _train_params(self) -> dict:
        """What training starts from. A built-in goes by name -- only the
        op's environment knows where Cellpose cached it."""
        params = {"collapse_channels": self.collapse_channels}
        if self.initial_model_name in BUILTIN_MODELS:
            params["model"] = PretrainedModel[self.initial_model_name]
        elif self.initial_model_name:
            params["initial_model"] = self._model_path(self.initial_model_name)
        return params

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "Cellpose4 (scikit-ops)", cls
        )
