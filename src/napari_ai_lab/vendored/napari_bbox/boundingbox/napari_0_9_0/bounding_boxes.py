"""What napari 0.9 requires of the bounding box *model*.

0.9's text drawing asks the layer which shapes are in view, through a
``_view_indices`` property that napari added on ``Shapes``. Without it,
adding the layer raises AttributeError from _vispy/utils/text.py.
"""

import numpy as np

from ..napari_0_8_0.bounding_boxes import (
    BoundingBoxLayer as BoundingBoxLayer_0_8_0,
)


class BoundingBoxLayer(BoundingBoxLayer_0_8_0):
    @property
    def _view_indices(self):
        return np.where(self._data_view._displayed)[0]
