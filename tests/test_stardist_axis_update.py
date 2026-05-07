"""
Test that changing StarDist inference_model_name automatically updates the axis.
"""

from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSegmenter import (
    StardistSegmenter,
)


def test_model_axis_recommendations():
    """Test that each model has the correct axis recommendation."""
    segmenter = StardistSegmenter()

    # Test 2D_versatile_fluo -> YX
    segmenter.inference_model_name = "2D_versatile_fluo"
    assert segmenter.get_recommended_axis() == "YX"

    # Test 2D_versatile_he -> YXC
    segmenter.inference_model_name = "2D_versatile_he"
    assert segmenter.get_recommended_axis() == "YXC"

    # Test 3D_demo -> ZYX
    segmenter.inference_model_name = "3D_demo"
    assert segmenter.get_recommended_axis() == "ZYX"


def test_model_axis_map():
    """Test that the model axis map is accessible."""
    segmenter = StardistSegmenter()
    axis_map = segmenter.get_model_axis_map()

    assert "2D_versatile_fluo" in axis_map
    assert "2D_versatile_he" in axis_map
    assert "3D_demo" in axis_map

    assert axis_map["2D_versatile_fluo"] == "YX"
    assert axis_map["2D_versatile_he"] == "YXC"
    assert axis_map["3D_demo"] == "ZYX"


def test_inference_model_name_change_prints_recommendation(capsys):
    """Test that changing inference_model_name prints the recommendation."""
    segmenter = StardistSegmenter()

    # Change to 3D model
    segmenter.inference_model_name = "3D_demo"

    # Check that something was printed
    captured = capsys.readouterr()
    assert "3D_demo" in captured.out
    assert "ZYX" in captured.out
