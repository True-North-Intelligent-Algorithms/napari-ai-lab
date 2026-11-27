"""Box annotation layers: the files, and the layers AI Lab makes of them.

A project's box layers are the folders under box_annotations/ -- runners,
bibs -- each with its own classes.yaml. The older layout, txt files directly
in box_annotations/, is the "Box annotations" layer.
"""

import numpy as np
import pytest
from skimage.io import imsave

from napari_ai_lab.utilities.box_annotations import (
    LEGACY,
    add_box_folder,
    box_file,
    box_folders,
    check_name,
    merge_class_names,
    prediction_file,
    prediction_sources,
    read_class_names,
    read_prediction_class_names,
    read_yolo_boxes,
    read_yolo_predictions,
    write_class_names,
    write_prediction_class_names,
    write_yolo_boxes,
)

BOX = np.array([[5.0, 5.0], [5.0, 25.0], [30.0, 25.0], [30.0, 5.0]])

# -- the files ----------------------------------------------------------------


def test_new_project_has_the_legacy_layer_only(tmp_path):
    assert box_folders(tmp_path) == [LEGACY]


def test_each_subfolder_is_a_layer(tmp_path):
    add_box_folder(tmp_path, "runners", first_class="runner")
    add_box_folder(tmp_path, "bibs", first_class="bib")
    assert box_folders(tmp_path) == ["bibs", "runners"]
    assert read_class_names(tmp_path, "runners") == {0: "runner"}
    assert read_class_names(tmp_path, "bibs") == {0: "bib"}
    # Each layer's classes are its own, not the project's.
    assert read_class_names(tmp_path) == {}


def test_legacy_files_stay_beside_new_layers(tmp_path):
    write_yolo_boxes(box_file(tmp_path, LEGACY, "a"), [], [], 10, 10)
    add_box_folder(tmp_path, "runners")
    assert box_folders(tmp_path) == [LEGACY, "runners"]


def test_legacy_classes_stay_at_the_project_root(tmp_path):
    write_class_names(tmp_path, {0: "buffalo"})
    assert (tmp_path / "classes.yaml").exists()
    assert read_class_names(tmp_path, LEGACY) == {0: "buffalo"}


def test_adding_an_existing_folder_keeps_its_classes(tmp_path):
    add_box_folder(tmp_path, "bibs", first_class="bib")
    write_class_names(tmp_path, {0: "bib", 1: "partial bib"}, "bibs")
    add_box_folder(tmp_path, "bibs", first_class="bib")
    assert read_class_names(tmp_path, "bibs") == {0: "bib", 1: "partial bib"}


@pytest.mark.parametrize("bad", ["", " ", "a/b", "..", "-x", "x.", "a:b"])
def test_bad_names_are_refused(bad):
    with pytest.raises(ValueError):
        check_name(bad)


def test_boxes_round_trip(tmp_path):
    box = np.array([[10.0, 20.0], [10.0, 60.0], [40.0, 60.0], [40.0, 20.0]])
    path = box_file(tmp_path, "runners", "IMG_1")
    write_yolo_boxes(path, [box], [2], 100, 200)
    boxes, classes = read_yolo_boxes(path, 100, 200)
    np.testing.assert_allclose(boxes[0], box, atol=1e-3)
    assert classes.tolist() == [2]


def test_predictions_keep_their_confidence(tmp_path):
    path = prediction_file(tmp_path, "TniaBibDetector", "IMG_1")
    write_yolo_boxes(
        path, [BOX, BOX], [0, 0], 40, 60, confidences=[0.41, None]
    )
    assert prediction_sources(tmp_path) == ["TniaBibDetector"]
    assert len(path.read_text().splitlines()[0].split()) == 6
    boxes, classes, conf = read_yolo_predictions(path, 40, 60)
    assert len(boxes) == 2
    assert conf[0] == pytest.approx(0.41)
    assert np.isnan(conf[1])  # no confidence: the column is left off


def test_classes_are_matched_by_name():
    mapping, merged = merge_class_names(
        {0: "cells_nuclei", 1: "cells", 2: "nuclei"}, {0: "nuclei", 1: "cells"}
    )
    # Same name, the layer's id; a new name, whose id 0 is taken, the next.
    assert mapping == {0: 2, 1: 1, 2: 0}
    assert merged == {0: "nuclei", 1: "cells", 2: "cells_nuclei"}


def test_a_new_name_whose_id_is_taken_gets_the_next_one():
    mapping, merged = merge_class_names({0: "bib"}, {0: "runner"})
    assert mapping == {0: 1}
    assert merged == {0: "runner", 1: "bib"}


def test_prediction_class_names_round_trip_and_go(tmp_path):
    write_prediction_class_names(tmp_path, "YoloSkopDetector", {1: "cells"})
    assert read_prediction_class_names(tmp_path, "YoloSkopDetector") == {
        1: "cells"
    }
    write_prediction_class_names(tmp_path, "YoloSkopDetector", None)
    assert read_prediction_class_names(tmp_path, "YoloSkopDetector") == {}


def test_a_trained_model_knows_its_class_names(tmp_path):
    from napari_ai_lab.Segmenters.GlobalSegmenters.SkopDetector import (
        trained_class_names,
    )

    # As train_yolo leaves its working files, beside the model.
    data = tmp_path / "cells_nuclei_training" / "dataset" / "data.yaml"
    data.parent.mkdir(parents=True)
    data.write_text(
        "path: 'D:\\images\\x'\ntrain: images/train\nval: images/val\n"
        "names:\n  0: 'cells_nuclei'\n  1: 'cells'\n  2: 'nuclei'\n"
    )
    weights = tmp_path / "cells_nuclei.pt"
    assert trained_class_names(weights) == {
        0: "cells_nuclei",
        1: "cells",
        2: "nuclei",
    }
    assert trained_class_names(tmp_path / "yolo11s.pt") == {}


class _FakeDetector:
    """What save_box_predictions reads from a detector after it ran."""

    def __init__(self, boxes, scores=None, classes=None, names=None):
        self.last_napari_boxes = boxes
        self.last_box_scores = scores
        self.last_box_classes = classes
        self.last_box_class_names = names


def test_model_saves_a_detectors_boxes(tmp_path):
    from napari_ai_lab.models import ImageDataModel

    imsave(tmp_path / "a.png", np.zeros((40, 60, 3), dtype=np.uint8))
    model = ImageDataModel(str(tmp_path))
    model.load_image(0)
    path = model.save_box_predictions(_FakeDetector([BOX], [0.9]), 0)
    assert path == prediction_file(tmp_path, "_FakeDetector", "a")
    boxes, _, conf = read_yolo_predictions(path, 40, 60)
    np.testing.assert_allclose(boxes[0], BOX, atol=1e-3)
    assert conf.tolist() == pytest.approx([0.9])
    # A segmenter that finds no boxes at all saves nothing.
    assert model.save_box_predictions(object(), 0) is None


def test_model_saves_a_detectors_classes_and_their_names(tmp_path):
    from napari_ai_lab.models import ImageDataModel

    imsave(tmp_path / "a.png", np.zeros((40, 60, 3), dtype=np.uint8))
    model = ImageDataModel(str(tmp_path))
    model.load_image(0)
    names = {1: "cells", 2: "nuclei"}
    path = model.save_box_predictions(
        _FakeDetector([BOX, BOX], [0.9, 0.8], [2, 1], names), 0
    )
    _, classes, _ = read_yolo_predictions(path, 40, 60)
    assert classes.tolist() == [2, 1]
    assert read_prediction_class_names(tmp_path, "_FakeDetector") == names

    # The next run's model names none: the old names go.
    model.save_box_predictions(_FakeDetector([BOX]), 0)
    assert read_prediction_class_names(tmp_path, "_FakeDetector") == {}


# -- the layers ---------------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    """Two small RGB photos, runners boxed in the first one only."""
    for name in ("a.png", "b.png"):
        imsave(tmp_path / name, np.zeros((40, 60, 3), dtype=np.uint8))
    add_box_folder(tmp_path, "runners", first_class="runner")
    write_yolo_boxes(box_file(tmp_path, "runners", "a"), [BOX], [0], 40, 60)
    # photo-enhance's bibs, found on the first photo only.
    write_yolo_boxes(
        prediction_file(tmp_path, "photo-enhance-bibs", "a"),
        [BOX, BOX],
        [0, 0],
        40,
        60,
        confidences=[0.8, 0.3],
    )
    return tmp_path


@pytest.fixture
def ai_lab(project, qtbot):
    import napari

    from napari_ai_lab.apps.nd_ai_lab_launcher import launch_nd_ai_lab

    viewer = napari.Viewer(show=False)
    widget, _, model = launch_nd_ai_lab(
        viewer,
        project,
        viewer_type="none",
        axes_to_collapse="C",
        axis_types="YXC",
        profile="2d-detect",
    )
    yield widget, viewer, model
    viewer.close()


def _go_to(widget, viewer, model, index):
    """Switch image the way the sequence viewer does."""
    layer = viewer.add_image(model.load_image(index), name="Image")
    widget._on_sequence_image_changed(layer, index)


def test_one_layer_per_folder_on_every_image(ai_lab):
    widget, viewer, model = ai_lab
    assert list(widget.box_layers) == ["runners"]
    runners = widget.box_layers["runners"]
    assert runners.name == "runners"
    assert len(runners.data) == 1

    # No file for the second image: the layer is there, and empty.
    _go_to(widget, viewer, model, 1)
    assert widget.box_layers["runners"] is runners
    assert len(runners.data) == 0


def test_added_layer_gets_a_folder_classes_and_saves(ai_lab, project):
    widget, viewer, model = ai_lab
    widget.new_box_layer_edit.setText("bibs")
    widget._on_add_box_layer()

    bibs = widget.box_layers["bibs"]
    assert viewer.layers.selection.active is bibs
    assert widget.shapes_layer is bibs
    assert read_class_names(project, "bibs") == {0: "bibs"}
    assert widget.class_combo.currentText() == "bibs"

    bibs.add_rectangles(
        [np.array([[1.0, 1.0], [1.0, 9.0], [9.0, 9.0], [9.0, 1.0]])]
    )
    widget._save_box_annotations()
    boxes, classes = read_yolo_boxes(box_file(project, "bibs", "a"), 40, 60)
    assert len(boxes) == 1
    # The runners layer, untouched, kept its box.
    boxes, _ = read_yolo_boxes(box_file(project, "runners", "a"), 40, 60)
    assert len(boxes) == 1


def test_class_combo_follows_the_selected_layer(ai_lab, project):
    widget, viewer, _ = ai_lab
    widget.new_box_layer_edit.setText("bibs")
    widget._on_add_box_layer()
    assert widget.class_combo.currentText() == "bibs"
    viewer.layers.selection.active = widget.box_layers["runners"]
    assert widget.class_combo.currentText() == "runner"


def test_deleted_layer_stays_deleted_and_keeps_its_files(ai_lab, project):
    widget, viewer, model = ai_lab
    viewer.layers.remove(widget.box_layers["runners"])
    _go_to(widget, viewer, model, 1)
    assert "runners" not in widget.box_layers
    assert box_file(project, "runners", "a").exists()


PREDICTED = "Box predictions (photo-enhance-bibs)"


def test_saved_predictions_show_on_their_image_only(ai_lab):
    widget, viewer, model = ai_lab
    layer = viewer.layers[PREDICTED]
    assert len(layer.data) == 2
    assert list(layer.features["score"]) == ["0.80", "0.30"]

    _go_to(widget, viewer, model, 1)
    assert viewer.layers[PREDICTED] is layer
    assert len(layer.data) == 0

    _go_to(widget, viewer, model, 0)
    assert len(layer.data) == 2


def test_deleted_prediction_layer_stays_deleted(ai_lab):
    widget, viewer, model = ai_lab
    viewer.layers.remove(PREDICTED)
    _go_to(widget, viewer, model, 1)
    _go_to(widget, viewer, model, 0)
    assert PREDICTED not in viewer.layers


def _copy(widget, source=PREDICTED, target="runners"):
    widget.copy_from_combo.setCurrentText(source)
    widget.copy_to_combo.setCurrentText(target)
    widget._on_copy_box_predictions()


def test_copy_row_lists_predictions_and_box_layers(ai_lab):
    widget, _, _ = ai_lab
    sources = [
        widget.copy_from_combo.itemText(i)
        for i in range(widget.copy_from_combo.count())
    ]
    targets = [
        widget.copy_to_combo.itemText(i)
        for i in range(widget.copy_to_combo.count())
    ]
    assert sources == [PREDICTED]
    assert targets == ["runners"]


def test_copy_adds_once_and_takes_the_targets_class(ai_lab, project):
    widget, viewer, _ = ai_lab
    add_box_folder(project, "bibs", first_class="bib")
    widget.new_box_layer_edit.setText("bibs")
    widget._on_add_box_layer()
    # Make runners the active layer, then copy to bibs: bibs' class 0.
    viewer.layers.selection.active = widget.box_layers["runners"]

    _copy(widget, target="bibs")
    bibs = widget.box_layers["bibs"]
    # The two predicted boxes are the same box, so one is copied.
    assert len(bibs.data) == 1
    assert list(bibs.features["name"]) == ["bib"]
    assert viewer.layers.selection.active is bibs

    _copy(widget, target="bibs")
    assert len(bibs.data) == 1  # copying again adds nothing


def test_copy_takes_only_the_selected_boxes(ai_lab):
    widget, viewer, _ = ai_lab
    predicted = viewer.layers[PREDICTED]
    other = np.array([[1.0, 40.0], [1.0, 55.0], [12.0, 55.0], [12.0, 40.0]])
    predicted.add_rectangles([other])
    predicted.selected_data = {2}

    runners = widget.box_layers["runners"]
    _copy(widget)
    assert len(runners.data) == 2  # its own box, plus the selected one
    np.testing.assert_allclose(runners.data[-1], other)


def test_running_a_detector_again_replaces_its_boxes(ai_lab):
    widget, viewer, _ = ai_lab
    segment = widget.segment_widget
    for _ in range(2):
        segment._add_predicted_boxes([BOX, BOX, BOX], (0, 0), "YXC", "Det")
    assert len(viewer.layers["Box predictions (Det)"].data) == 3


# -- a model that names its classes -------------------------------------------

CELLS = {0: "cells_nuclei", 1: "cells", 2: "nuclei"}
OTHER = np.array([[1.0, 40.0], [1.0, 55.0], [12.0, 55.0], [12.0, 40.0]])
NAMED = "Box predictions (YoloSkopDetector)"


def _named_predictions(project, widget, viewer, model):
    """A YOLO model's cell and nucleus on the first photo, a cell on the
    second, saved as a run saves them."""
    write_yolo_boxes(
        prediction_file(project, "YoloSkopDetector", "a"),
        [BOX, OTHER],
        [1, 2],
        40,
        60,
        confidences=[0.8, 0.6],
    )
    write_yolo_boxes(
        prediction_file(project, "YoloSkopDetector", "b"), [OTHER], [1], 40, 60
    )
    write_prediction_class_names(project, "YoloSkopDetector", CELLS)
    _go_to(widget, viewer, model, 0)
    return viewer.layers[NAMED]


def test_named_predictions_show_their_class_in_its_colour(ai_lab, project):
    widget, viewer, model = ai_lab
    layer = _named_predictions(project, widget, viewer, model)
    assert list(layer.features["label"]) == ["cells 0.80", "nuclei 0.60"]
    assert not np.allclose(layer.edge_color[0], layer.edge_color[1])
    # Deleting one works: the text colours are in step.
    layer.selected_data = {0}
    layer.remove_selected()
    assert list(layer.features["label"]) == ["nuclei 0.60"]
    # Unnamed ones show their score only, in one colour.
    plain = viewer.layers[PREDICTED]
    assert list(plain.features["label"]) == ["0.80", "0.30"]
    assert np.allclose(plain.edge_color[0], plain.edge_color[1])


def test_a_detector_run_shows_its_named_classes(ai_lab):
    widget, viewer, _ = ai_lab
    widget.segment_widget._add_predicted_boxes(
        [BOX, OTHER], (0, 0), "YXC", "Det", [0.9, 0.5], [2, 1], CELLS
    )
    layer = viewer.layers["Box predictions (Det)"]
    assert list(layer.features["label"]) == ["nuclei 0.90", "cells 0.50"]


def test_copy_keeps_named_classes_adding_names_the_layer_lacks(
    ai_lab, project
):
    widget, viewer, model = ai_lab
    _named_predictions(project, widget, viewer, model)
    _copy(widget, source=NAMED)
    runners = widget.box_layers["runners"]
    # BOX is already there; OTHER, a nucleus, is new to runners.
    assert list(runners.features["name"]) == ["runner", "nuclei"]
    assert list(runners.features["class"]) == [0, 2]
    assert read_class_names(project, "runners") == {0: "runner", 2: "nuclei"}
    assert widget.class_combo.currentText() == "runner"  # still the next box's
    assert runners.feature_defaults["class"][0] == 0


def test_copy_to_all_photos_keeps_named_classes(ai_lab, project):
    widget, viewer, model = ai_lab
    _named_predictions(project, widget, viewer, model)
    widget._ask_copy_all = lambda text: "copy"
    widget.copy_from_combo.setCurrentText(NAMED)
    widget.copy_to_combo.setCurrentText("runners")
    widget._on_copy_box_predictions_all()
    # Photo a has runners boxes already, so only b's cell is copied.
    _, classes = read_yolo_boxes(box_file(project, "runners", "b"), 40, 60)
    assert classes.tolist() == [1]
    assert read_class_names(project, "runners") == {0: "runner", 1: "cells"}


# -- changing a box's class ---------------------------------------------------


def _add_nucleus_class(widget):
    """Type a second class, "nucleus", into the class combo: class 1."""
    widget.box_layers["runners"].selected_data = set()
    widget.class_combo.lineEdit().setText("nucleus")
    widget._on_class_typed()
    assert widget.class_combo.currentData() == 1


def test_ctrl_number_gives_the_selected_boxes_that_class(ai_lab):
    from app_model.types import KeyBinding

    widget, _, _ = ai_lab
    _add_nucleus_class(widget)
    runners = widget.box_layers["runners"]
    runners.selected_data = {0}

    runners.keymap[KeyBinding.from_str("Ctrl+1")](runners)
    assert list(runners.features["class"]) == [0]
    runners.keymap[KeyBinding.from_str("Ctrl+2")](runners)
    assert list(runners.features["class"]) == [1]
    assert list(runners.features["name"]) == ["nucleus"]


def test_the_class_menu_lists_the_classes_and_sets_them(ai_lab):
    widget, _, _ = ai_lab
    _add_nucleus_class(widget)
    runners = widget.box_layers["runners"]
    runners.selected_data = {0}

    menu = widget._class_menu()
    assert [a.text() for a in menu.actions()] == [
        "runner\tCtrl+1",
        "nucleus\tCtrl+2",
    ]
    menu.actions()[1].trigger()
    assert list(runners.features["class"]) == [1]


class _Click:
    """What napari's mouse event tells a drag callback, for a right-click."""

    button = 2
    type = "mouse_press"

    def __init__(self, position, pos=(100.0, 100.0)):
        self.position = position
        self.pos = pos


def test_right_click_on_a_box_selects_it_and_opens_the_menu(
    ai_lab, monkeypatch, qtbot
):
    widget, _, _ = ai_lab
    runners = widget.box_layers["runners"]
    shown = []
    monkeypatch.setattr(widget, "_show_class_menu", lambda: shown.append(1))

    click = _Click(position=(15.0, 15.0))  # inside BOX
    gen = widget._on_box_right_click(runners, click)
    next(gen)
    click.type = "mouse_release"
    with pytest.raises(StopIteration):
        next(gen)
    assert runners.selected_data == {0}
    qtbot.waitUntil(lambda: bool(shown), timeout=1000)


def test_right_drag_or_a_click_off_the_boxes_opens_no_menu(ai_lab):
    widget, _, _ = ai_lab
    runners = widget.box_layers["runners"]

    # Off every box: not even a drag callback.
    assert (
        next(widget._on_box_right_click(runners, _Click((38.0, 58.0))), 0) == 0
    )

    click = _Click(position=(15.0, 15.0))
    gen = widget._on_box_right_click(runners, click)
    next(gen)
    click.type, click.pos = "mouse_release", (160.0, 100.0)  # dragged: zoom
    with pytest.raises(StopIteration):
        next(gen)
    assert runners.selected_data == set()
