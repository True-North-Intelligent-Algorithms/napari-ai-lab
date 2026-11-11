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
    prediction_file,
    prediction_sources,
    read_class_names,
    read_yolo_boxes,
    read_yolo_predictions,
    write_class_names,
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


class _FakeDetector:
    """What save_box_predictions reads from a detector after it ran."""

    def __init__(self, boxes, scores=None):
        self.last_napari_boxes = boxes
        self.last_box_scores = scores


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
