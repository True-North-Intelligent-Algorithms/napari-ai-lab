"""Test TrainingBase mixin functionality."""

from pathlib import Path

from napari_ai_lab.Segmenters.GlobalSegmenters.MonaiUNetSegmenter import (
    MonaiUNetSegmenter,
)


def test_training_base_initialization():
    """Test that TrainingBase initializes loss tracking lists."""
    segmenter = MonaiUNetSegmenter()

    # Check loss tracking lists are initialized
    assert hasattr(segmenter, "train_loss_list")
    assert hasattr(segmenter, "validation_loss_list")
    assert isinstance(segmenter.train_loss_list, list)
    assert isinstance(segmenter.validation_loss_list, list)
    assert len(segmenter.train_loss_list) == 0
    assert len(segmenter.validation_loss_list) == 0


def test_patch_path_property():
    """Test that patch_path property works correctly."""
    segmenter = MonaiUNetSegmenter()

    # Initially should be None
    assert segmenter.patch_path is None

    # Should be able to set it
    test_path = "/path/to/patches"
    segmenter.patch_path = test_path
    assert segmenter.patch_path == test_path

    # Should be able to set it to a Path object
    path_obj = Path("/another/path")
    segmenter.patch_path = path_obj
    assert segmenter.patch_path == path_obj


def test_train_method_exists():
    """Test that train method exists and is callable."""
    segmenter = MonaiUNetSegmenter()

    assert hasattr(segmenter, "train")
    assert callable(segmenter.train)


def test_train_with_patch_path():
    """Test that train method can access patch_path."""
    segmenter = MonaiUNetSegmenter()
    segmenter.patch_path = "/test/patch/path"

    # Call train (will return not implemented message)
    result = segmenter.train()

    # Should return a dict
    assert isinstance(result, dict)
    assert "success" in result


if __name__ == "__main__":
    test_training_base_initialization()
    test_patch_path_property()
    test_train_method_exists()
    test_train_with_patch_path()
    print("✅ All tests passed!")
