"""
Batch segmentation over a sequence of images (spec 0006).

The folder-of-tiffs case: a directory of RGB images processed one after the
other, each saved as it is produced, with images the segmenter cannot handle
skipped rather than stopping the run.

No Qt and no viewer here -- headless is just a call with no callbacks.
"""

import numpy as np
import pytest
import tifffile

from napari_ai_lab.models.image_data_model import ImageDataModel
from napari_ai_lab.Segmenters.SegmenterBase import SegmenterBase

NUM_RGB = 12
SHAPE = (32, 32, 3)


class CountingSegmenter(SegmenterBase):
    """Returns a constant mask and records every slice it was handed."""

    def __init__(self):
        super().__init__()
        self.supported_axes = ["YX"]
        self.seen = []

    def segment(
        self, image_slice, points=None, shapes=None, parent_directory=None
    ):
        self.seen.append(image_slice.shape)
        return np.ones(image_slice.shape[:2], dtype=np.uint16)


@pytest.fixture
def sequence_dir(tmp_path):
    """Twelve RGB images, plus one greyscale one that cannot be segmented."""
    for i in range(NUM_RGB):
        tifffile.imwrite(
            tmp_path / f"img_{i:03d}.tif", np.zeros(SHAPE, np.uint8)
        )
    tifffile.imwrite(
        tmp_path / f"img_{NUM_RGB:03d}.tif", np.zeros((32, 32), np.uint8)
    )
    return tmp_path


def test_sequence_is_processed_saved_and_skipped(sequence_dir):
    """Every RGB image is segmented and saved; the greyscale one is skipped."""
    model = ImageDataModel(str(sequence_dir))
    model.load_image(0)
    segmenter = CountingSegmenter()

    processor = model.segment_sequence(
        segmenter=segmenter,
        selected_axis="YX",
        axes_to_collapse=["C"],
        current_index=0,
    )

    assert processor.processed == list(range(NUM_RGB))
    assert len(processor.skipped) == 1
    index, reason = processor.skipped[0]
    assert index == NUM_RGB
    assert "no C to collapse" in reason

    # One slice per image, each the whole RGB plane -- not one per channel.
    assert segmenter.seen == [SHAPE] * NUM_RGB

    saved = sorted(
        model.get_predictions_directory("CountingSegmenter").glob("*.tif")
    )
    assert [p.stem for p in saved] == [f"img_{i:03d}" for i in range(NUM_RGB)]


def test_model_is_left_on_the_viewed_image(sequence_dir):
    """The batch ends with the model back where the viewer is."""
    model = ImageDataModel(str(sequence_dir))
    model.load_image(0)

    model.segment_sequence(
        segmenter=CountingSegmenter(),
        selected_axis="YX",
        axes_to_collapse=["C"],
        current_index=0,
    )

    assert model.image_data.shape == SHAPE
    assert model.axis_types == "YXC"


def test_range_of_images(sequence_dir):
    """start_index and end_index count images, not slices."""
    model = ImageDataModel(str(sequence_dir))
    model.load_image(2)

    processor = model.segment_sequence(
        segmenter=CountingSegmenter(),
        selected_axis="YX",
        axes_to_collapse=["C"],
        start_index=2,
        end_index=4,
        current_index=2,
    )

    assert processor.processed == [2, 3, 4]
    assert processor.skipped == []
