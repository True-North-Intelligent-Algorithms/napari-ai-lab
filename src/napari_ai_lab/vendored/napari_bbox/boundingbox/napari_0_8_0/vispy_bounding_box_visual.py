"""What napari 0.8 requires of the bounding box *visual node*.

``ClippingPlanesMixin.__init__`` gained a required keyword-only ``font_info``
argument, so constructing the compound visual without it fails:

    TypeError: ClippingPlanesMixin.__init__() missing 1 required keyword-only
    argument: 'font_info'

Taking ``**kwargs`` rather than naming ``font_info`` means whatever the next
release adds passes through as well.
"""

from vispy.scene.visuals import Line, Markers, Mesh, Text

from ..napari_0_5_0.vispy_bounding_box_visual import (
    BoundingBoxVisual as BoundingBoxVisual_0_5_0,
)


class BoundingBoxVisual(BoundingBoxVisual_0_5_0):
    def __init__(self, **kwargs) -> None:
        # Skips the 0.5.0 __init__, which builds the same subvisuals but passes
        # nothing through to ClippingPlanesMixin.
        super(BoundingBoxVisual_0_5_0, self).__init__(
            [Mesh(), Mesh(), Line(), Markers(), Text()], **kwargs
        )
