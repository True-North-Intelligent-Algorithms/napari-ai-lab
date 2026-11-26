"""Box layers: a colour per class, and small boxes on top of big ones."""

import napari
import numpy as np

from napari_ai_lab.apps.nd_ai_lab import _restyle_on_edit, _style_boxes


def _box(y0, x0, y1, x1):
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], float)


def _box_layer():
    """A box layer as NDAILab._add_box_layer makes one, minus the viewer."""
    layer = napari.layers.Shapes(
        ndim=2,
        edge_color="yellow",
        face_color="transparent",
        features={
            "class": np.zeros(0, dtype=int),
            "name": np.zeros(0, dtype=object),
        },
        feature_defaults={"class": 0, "name": "nucleus"},
        text={"string": "{name}", "size": 12, "color": "yellow"},
        metadata={"box_folder": "cells", "box_colour": "yellow"},
    )
    layer.events.data.connect(_restyle_on_edit)
    return layer


def _colour(name):
    return napari.utils.colormaps.standardize_color.transform_color(name)[0]


def test_a_small_box_drawn_first_is_picked_inside_a_big_one():
    layer = _box_layer()
    layer.add_rectangles([_box(40, 40, 50, 50)])
    layer.feature_defaults = {"class": 1, "name": "cell"}
    layer.add_rectangles([_box(0, 0, 100, 100)])  # on top, as drawn later

    assert layer._data_view.inside([45, 45]) == 0  # the small one
    assert layer._data_view.inside([10, 10]) == 1


def test_each_class_has_its_own_colour_text_too():
    layer = _box_layer()
    layer.add_rectangles([_box(0, 0, 100, 100)])
    layer.feature_defaults = {"class": 1, "name": "cell"}
    layer.add_rectangles([_box(40, 40, 50, 50)])

    yellow, blue = _colour("yellow"), _colour("deepskyblue")
    np.testing.assert_allclose(layer.edge_color, [yellow, blue])
    np.testing.assert_allclose(
        layer.text.color(layer.features), [yellow, blue]
    )


def test_a_class_changed_is_recoloured():
    layer = _box_layer()
    layer.add_rectangles([_box(0, 0, 10, 10), _box(20, 20, 30, 30)])
    features = layer.features.copy()
    features.loc[[1], "class"] = 2
    layer.features = features
    _style_boxes(layer)

    np.testing.assert_allclose(
        layer.edge_color, [_colour("yellow"), _colour("violet")]
    )


def test_a_box_moved_bigger_goes_to_the_back():
    layer = _box_layer()
    layer.add_rectangles([_box(0, 0, 10, 10), _box(40, 40, 50, 50)])
    layer.selected_data = {0}
    layer.data = [_box(0, 0, 100, 100), layer.data[1]]

    assert layer._data_view.inside([45, 45]) == 1
