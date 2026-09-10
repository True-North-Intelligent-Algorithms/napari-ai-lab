"""Cellpose 3 (CPnet) as a scikit-ops segmenter, with a model combo.

Nine built-ins, each trained on one imaging domain. Channels are named
rather than numbered -- Cellpose 3's numbering is easy to transpose.
"""

from dataclasses import dataclass, field

from .CellposeSkopSegmenterBase import CellposeSkopSegmenterBase
from .GlobalSegmenterBase import GlobalSegmenterBase

try:
    from skop.ops.segment.cellpose3 import (
        CytoplasmChannel,
        NucleusChannel,
        PretrainedModel,
        cellpose3,
    )
except ImportError:
    cellpose3 = CytoplasmChannel = NucleusChannel = PretrainedModel = None

try:
    # Separate: a scikit-ops without the training op loses the Train button,
    # not the segmenter.
    from skop.ops.train import train_cellpose3
except ImportError:
    train_cellpose3 = None


#: Built-in -> axes. All accept colour and collapse it when asked for grey.
BUILTIN_MODELS = {
    "cyto3": "YXC",
    "nuclei": "YXC",
    "bacteria_phase": "YXC",
    "bacteria_fluorescence": "YXC",
    "yeast_phase": "YXC",
    "yeast_brightfield": "YXC",
    "tissue": "YXC",
    "live_cell": "YXC",
    "deepbacs": "YXC",
}


@dataclass
class Cellpose3SkopSegmenter(CellposeSkopSegmenterBase):
    """Cellpose 3 through scikit-ops, with the model chosen by combo."""

    #: grayscale, red, green or blue. Text, not a combo: the form generator
    #: has no choice widget yet.
    cytoplasm_channel: str = field(
        default="grayscale",
        metadata={
            "type": "str",
            "param_type": "inference",
            "default": "grayscale",
        },
    )
    #: Read only when the cytoplasm channel names a colour too.
    nucleus_channel: str = field(
        default="none",
        metadata={
            "type": "str",
            "param_type": "inference",
            "default": "none",
        },
    )

    BUILTIN_MODELS = BUILTIN_MODELS
    FLAVOR = "cellpose3"
    # staticmethod, not the bare function: a function stored as a class
    # attribute is a descriptor, so `self.TRAIN_OP` would hand back a *bound
    # method*, and skop then fails to read its spec off that. Same reason
    # SkopSegmenter.register_op wraps the op it is given.
    TRAIN_OP = staticmethod(train_cellpose3)

    instructions = """
Cellpose 3:
• diameter - low finds small, high finds big. Most important here, it
  resizes the image to match.
• cell probability - -6 to 6, lower finds more
• flow threshold - 0 to 3, higher finds more, 0 turns it off (finds more)
• iters - higher runs more flow iterations, finds larger
• model - cyto3 is the generalist, the rest are trained on one kind of image
• channels - grayscale averages the colours, or name the colour the objects
  are in
• training needs about 4 GB GPU
    """

    @property
    def op(self):
        return cellpose3

    def _model_params(self) -> dict:
        """A built-in by name, or a checkpoint by path -- never both."""
        params = {
            "cytoplasm_channel": CytoplasmChannel(self.cytoplasm_channel),
            "nucleus_channel": NucleusChannel(self.nucleus_channel),
        }
        if self.inference_model_name in BUILTIN_MODELS:
            params["model"] = PretrainedModel[self.inference_model_name]
        else:
            params["pretrained_model"] = self._model_path(
                self.inference_model_name
            )
        return params

    def _train_params(self) -> dict:
        """What training starts from, and the channels. A built-in goes by
        name -- only the op's environment knows where Cellpose cached it."""
        params = {
            "cytoplasm_channel": CytoplasmChannel(self.cytoplasm_channel),
            "nucleus_channel": NucleusChannel(self.nucleus_channel),
        }
        if self.initial_model_name in BUILTIN_MODELS:
            params["model"] = PretrainedModel[self.initial_model_name]
        elif self.initial_model_name:
            params["initial_model"] = self._model_path(self.initial_model_name)
        return params

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "Cellpose3 (scikit-ops)", cls
        )
