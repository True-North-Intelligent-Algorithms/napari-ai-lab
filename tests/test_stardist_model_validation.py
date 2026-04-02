"""
Test that StarDist model validation works correctly.
"""

from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSegmenter import (
    StardistSegmenter,
)


def test_recommended_axis_for_each_model():
    """Test that each model returns the correct recommended axis."""
    segmenter = StardistSegmenter()

    # Test 2D_versatile_fluo -> YX
    segmenter.model_preset = "2D_versatile_fluo"
    assert segmenter.get_recommended_axis() == "YX"

    # Test 2D_versatile_he -> YXC
    segmenter.model_preset = "2D_versatile_he"
    assert segmenter.get_recommended_axis() == "YXC"

    # Test 3D_demo -> ZYX
    segmenter.model_preset = "3D_demo"
    assert segmenter.get_recommended_axis() == "ZYX"


def test_potential_axes_include_all():
    """Test that potential axes include all model requirements."""
    segmenter = StardistSegmenter()

    # All these axes should be in potential_axes
    assert "YX" in segmenter.potential_axes
    assert "YXC" in segmenter.potential_axes
    assert "ZYX" in segmenter.potential_axes
    assert "ZYXC" in segmenter.potential_axes


def test_supported_axes_filtering():
    """Test that supported_axes can be filtered based on image."""
    segmenter = StardistSegmenter()

    # Simulate filtering for a grayscale 2D image (no C, no Z)
    segmenter.supported_axes = ["YX"]

    # Model requiring C should not be in supported axes
    segmenter.model_preset = "2D_versatile_he"
    recommended = segmenter.get_recommended_axis()
    assert recommended == "YXC"
    assert (
        recommended not in segmenter.supported_axes
    )  # Should not be compatible

    # Model not requiring C should be in supported axes
    segmenter.model_preset = "2D_versatile_fluo"
    recommended = segmenter.get_recommended_axis()
    assert recommended == "YX"
    assert recommended in segmenter.supported_axes  # Should be compatible


def test_model_preset_change_message(capsys):
    """Test that changing model_preset triggers the debug message."""
    segmenter = StardistSegmenter()

    # Initial value
    segmenter.model_preset = "2D_versatile_fluo"

    # Change to 3D
    segmenter.model_preset = "3D_demo"

    # Check output
    captured = capsys.readouterr()
    assert "3D_demo" in captured.out
    assert "ZYX" in captured.out
