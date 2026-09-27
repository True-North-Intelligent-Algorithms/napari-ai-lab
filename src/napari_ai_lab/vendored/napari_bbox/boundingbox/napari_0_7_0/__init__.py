# Order matters. The key bindings module has no __all__ and imports
# BoundingBoxLayer itself, so a wildcard import of it re-exports the older
# class and silently overwrites ours. It therefore goes first, and the
# explicit imports below win.
from ._bounding_boxes_key_bindings import *  # noqa: F401,F403

from .bounding_boxes import BoundingBoxLayer
from .qt_bounding_box_control import register_layer_control
from .vispy_bounding_box_layer import register_layer_visual

__all__ = [
    "BoundingBoxLayer",
    "register_layer_control",
    "register_layer_visual",
]
