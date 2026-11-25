import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_ai_lab.apps.nd_easy_segment import NDEasySegment
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    MonaiUNetSegmenter,
    ThresholdSegmenter,
)

#: MONAI's trained test model. Test data is not in git (tests/test_images is
#: ignored), so this exists only where someone put it.
MONAI_MODEL = (
    Path(__file__).parent
    / "test_images"
    / "vessels_project"
    / "models"
    / "monai_unet_test.pth"
)

#: The widget path is the same for any segmenter. ThresholdSegmenter needs
#: nothing optional and no model, so the path is tested everywhere, CI
#: included; MONAI where it and its test model are present.
SEGMENTERS = [
    pytest.param(ThresholdSegmenter, "ThresholdSegmenter", id="threshold"),
    pytest.param(
        MonaiUNetSegmenter,
        "MonaiUNetSegmenter",
        id="monai",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("monai") is None
            or not MONAI_MODEL.exists(),
            reason=f"needs MONAI and its test model, {MONAI_MODEL}",
        ),
    ),
]


def _stack(path: Path):
    """A small ZYX stack with bright blobs, written as a project's image."""
    rng = np.random.default_rng(0)
    stack = rng.normal(100, 5, (7, 96, 128)).clip(0).astype(np.uint16)
    yy, xx = np.mgrid[:96, :128]
    for cy, cx in ((30, 40), (60, 90), (70, 30)):
        stack[:, (yy - cy) ** 2 + (xx - cx) ** 2 < 100] += 400
    tifffile.imwrite(path, stack)


@pytest.mark.parametrize(("segmenter_class", "segmenter_name"), SEGMENTERS)
def test_nd_easy_segment_creates_prediction(
    qtbot, tmp_path, segmenter_class, segmenter_name
):
    """Segment the current slice from the widget; a prediction file appears."""
    import napari

    segmenter_class.register()
    _stack(tmp_path / "stack.tif")

    model = ImageDataModel(str(tmp_path))
    model.set_prediction_io_type("tiff_slice", axis_slice="YX")

    viewer = napari.Viewer(show=False)
    widget = NDEasySegment(viewer, model)
    widget.automatic_mode_btn.setChecked(True)

    widget.segmenter_combo.setCurrentText(segmenter_name)
    segmenter = widget.segmenter_cache[segmenter_name]
    if segmenter_class is MonaiUNetSegmenter:
        segmenter.load_model(str(MONAI_MODEL))
    widget._update_segmenter_parameter_form(segmenter)
    widget.segmenter_parameter_form.set_selected_axis("YX")

    image_layer = viewer.add_image(model.load_image(0), name="Image")
    widget._set_image_layer(image_layer)

    # It runs on a worker thread, so wait for its result to come back.
    widget._on_segment_current()
    qtbot.waitUntil(
        lambda: widget._segment_current_thread is None, timeout=120_000
    )

    predictions_dir = model.get_predictions_directory(algorithm=segmenter_name)
    # napari opens a stack on its middle plane: 3 of 0-6.
    prediction_file = predictions_dir / "stack_3.tif"
    assert prediction_file.exists(), (
        f"Prediction file not found: {prediction_file}; "
        f"there: {sorted(p.name for p in predictions_dir.glob('*'))}"
    )
    # Not viewer.close(): on a viewer never shown, napari's closeEvent
    # raises in its own dims animation thread ("wrapped C/C++ object of type
    # AnimationThread has been deleted"), which pytest-qt fails the test on.
