"""What napari 0.8 requires of the bounding box *model*.

napari 0.8 moved layer slicing behind a state object: ``Layer`` gained a new
``@abstractmethod``, ``_get_layer_slicing_state``. Because it is abstract,
adding it is a breaking change for every existing subclass -- the layer cannot
be instantiated at all until it is implemented:

    TypeError: Can't instantiate abstract class BoundingBoxLayer without an
    implementation for abstract method '_get_layer_slicing_state'

The implementation is a thin adapter that routes back to the layer's own
``_set_view_slice``, which is exactly what napari's own ``Shapes`` does.
"""

from napari.layers.base.base import _LayerSlicingState

from ..napari_0_6_0.bounding_boxes import (
    BoundingBoxLayer as BoundingBoxLayer_0_6_0,
)


class _BoundingBoxSlicingState(_LayerSlicingState):
    """Routes napari >= 0.8 slicing back to the layer's own _set_view_slice."""

    def _set_view_slice(self):
        self.layer._set_view_slice()


class BoundingBoxLayer(BoundingBoxLayer_0_6_0):
    def _get_layer_slicing_state(self, data, cache):
        return _BoundingBoxSlicingState(layer=self, data=data, cache=cache)
