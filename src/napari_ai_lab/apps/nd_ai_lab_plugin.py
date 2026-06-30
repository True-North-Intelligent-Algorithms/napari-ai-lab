"""
napari-menu entry point for ND AI Lab.

When launched from the napari plugin menu we want the full set of
augmenters and segmenters available, so this thin subclass registers
everything before constructing NDAILab.

Scripts that want selective registration use ``launch_nd_ai_lab.py``
and instantiate :class:`NDAILab` directly.

Why a subclass and not a factory function?  Napari's plugin loader only
performs viewer-injection for **class** widget contributions; plain
functions are assumed to be magicgui widgets and receive no viewer.
"""

from napari.viewer import Viewer

from .nd_ai_lab import NDAILab
from .register_all import register_all


class NDAILabPlugin(NDAILab):
    """NDAILab variant that registers every segmenter/augmenter on launch."""

    def __init__(self, viewer: Viewer):
        register_all()
        super().__init__(viewer)
