# Order matters, as in napari_0_8_0: the key bindings module re-exports the
# older class through its wildcard, so it goes first.
from ..napari_0_8_0._bounding_boxes_key_bindings import *  # noqa: F401,F403

from ..napari_0_8_0.qt_bounding_box_control import register_layer_control
from ..napari_0_8_0.vispy_bounding_box_layer import register_layer_visual
from .bounding_boxes import BoundingBoxLayer

__all__ = [
    "BoundingBoxLayer",
    "register_layer_control",
    "register_layer_visual",
]
