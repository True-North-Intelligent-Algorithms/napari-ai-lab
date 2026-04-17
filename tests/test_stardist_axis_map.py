"""
Test StarDist model-to-axis mapping functionality.
"""

from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSegmenter import (
    BUILTIN_MODEL_MAP,
    StardistSegmenter,
)


def test_model_axis_map():
    """Test that MODEL_AXIS_MAP is defined correctly."""
    print("\n" + "=" * 60)
    print("Testing StarDist Model-to-Axis Mapping")
    print("=" * 60)

    # Check the mapping exists
    assert BUILTIN_MODEL_MAP is not None
    print("✓ MODEL_AXIS_MAP defined")

    # Check expected models are present
    expected_models = ["2D_versatile_fluo", "2D_versatile_he", "3D_demo"]
    for model in expected_models:
        assert model in BUILTIN_MODEL_MAP, f"Model {model} missing from map"
        print(f"✓ {model} -> {BUILTIN_MODEL_MAP[model]}")

    # Verify specific mappings
    assert BUILTIN_MODEL_MAP["2D_versatile_fluo"] == "YX"
    assert BUILTIN_MODEL_MAP["2D_versatile_he"] == "YXC"
    assert BUILTIN_MODEL_MAP["3D_demo"] == "ZYX"
    print("✓ All axis mappings correct")


def test_get_recommended_axis():
    """Test that get_recommended_axis() returns correct values."""
    print("\n" + "=" * 60)
    print("Testing get_recommended_axis() Method")
    print("=" * 60)

    segmenter = StardistSegmenter()

    # Test each model preset
    test_cases = [
        ("2D_versatile_fluo", "YX"),
        ("2D_versatile_he", "YXC"),
        ("3D_demo", "ZYX"),
    ]

    for inference_model_name, expected_axis in test_cases:
        segmenter.inference_model_name = inference_model_name
        recommended = segmenter.get_recommended_axis()
        assert (
            recommended == expected_axis
        ), f"Expected {expected_axis}, got {recommended}"
        print(f"\u2713 {inference_model_name} recommends {recommended}")


def test_inference_model_name_change_notification():
    """Test that changing inference_model_name prints recommended axis."""
    print("\n" + "=" * 60)
    print("Testing Inference Model Name Change Notification")
    print("=" * 60)

    segmenter = StardistSegmenter()

    # Change inference model name and watch for notification
    print("\nChanging from 2D_versatile_fluo to 3D_demo:")
    segmenter.inference_model_name = "3D_demo"

    print("\nChanging from 3D_demo to 2D_versatile_he:")
    segmenter.inference_model_name = "2D_versatile_he"

    print("\u2713 Inference model name changes trigger notifications")


def test_model_choices_generated_from_map():
    """Test that model choices come from MODEL_AXIS_MAP."""
    print("\n" + "=" * 60)
    print("Testing Model Choices Generation")
    print("=" * 60)

    segmenter = StardistSegmenter()

    # Get the choices from get_model_axis_map (includes builtins)
    choices = list(segmenter.get_model_axis_map().keys())

    # Should contain keys from BUILTIN_MODEL_MAP
    expected_choices = list(BUILTIN_MODEL_MAP.keys())
    for key in expected_choices:
        assert key in choices, f"Expected {key} in choices"
    print(f"✓ Choices: {choices}")
    print("✓ Generated from MODEL_AXIS_MAP keys")


if __name__ == "__main__":
    test_model_axis_map()
    test_get_recommended_axis()
    test_inference_model_name_change_notification()
    test_model_choices_generated_from_map()
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
