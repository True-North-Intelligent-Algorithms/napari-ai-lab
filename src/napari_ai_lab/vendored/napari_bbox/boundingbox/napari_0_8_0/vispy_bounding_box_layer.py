"""What napari 0.8 requires of the bounding box *visual layer*.

``VispyBaseLayer.__init__`` gained a required ``font_info`` argument, which the
Qt viewer passes when it builds the visual for every layer:

    TypeError: VispyBoundingBoxLayer.__init__() got an unexpected keyword
    argument 'font_info'

napari's own ``VispyShapesLayer`` passes ``font_info`` to both the visual and
the base layer, and this mirrors that. ``**kwargs`` rather than a named
``font_info`` so the next added argument needs no further patch.
"""

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.utils.visual import layer_to_visual

from ..napari_0_5_0.vispy_bounding_box_layer import (
    VispyBoundingBoxLayer as VispyBoundingBoxLayer_0_5_0,
)
from .vispy_bounding_box_visual import BoundingBoxVisual


class VispyBoundingBoxLayer(VispyBoundingBoxLayer_0_5_0):
    def __init__(self, layer, **kwargs) -> None:
        node = BoundingBoxVisual(**kwargs)
        # Skips the 0.5.0 __init__, which builds a visual without font_info and
        # calls VispyBaseLayer.__init__ without it either. Everything after the
        # constructor is unchanged, so it is repeated here rather than
        # refactored upstream in the vendored copy.
        VispyBaseLayer.__init__(self, layer, node, **kwargs)

        self.layer.events.edge_width.connect(self._on_data_change)
        self.layer.events.edge_color.connect(self._on_data_change)
        self.layer.events.face_color.connect(self._on_data_change)
        self.layer.text.events.connect(self._on_text_change)
        self.layer.events.highlight.connect(self._on_highlight_change)

        # TODO: move to overlays
        self.node.highlight_vertices.symbol = "square"
        self.node.highlight_vertices.scaling = False

        self.reset()
        self._on_data_change()


def register_layer_visual(layer_type):
    layer_to_visual[layer_type] = VispyBoundingBoxLayer
