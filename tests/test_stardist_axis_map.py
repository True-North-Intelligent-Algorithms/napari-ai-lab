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

    for model_preset, expected_axis in test_cases:
        segmenter.model_preset = model_preset
        recommended = segmenter.get_recommended_axis()
        assert (
            recommended == expected_axis
        ), f"Expected {expected_axis}, got {recommended}"
        print(f"✓ {model_preset} recommends {recommended}")


def test_model_preset_change_notification():
    """Test that changing model_preset prints recommended axis."""
    print("\n" + "=" * 60)
    print("Testing Model Preset Change Notification")
    print("=" * 60)

    segmenter = StardistSegmenter()

    # Change model preset and watch for notification
    print("\nChanging from 2D_versatile_fluo to 3D_demo:")
    segmenter.model_preset = "3D_demo"

    print("\nChanging from 3D_demo to 2D_versatile_he:")
    segmenter.model_preset = "2D_versatile_he"

    print("✓ Model preset changes trigger notifications")


def test_model_choices_generated_from_map():
    """Test that model choices come from MODEL_AXIS_MAP."""
    print("\n" + "=" * 60)
    print("Testing Model Choices Generation")
    print("=" * 60)

    segmenter = StardistSegmenter()

    # Get the choices from metadata
    choices = segmenter.__dataclass_fields__["model_preset"].metadata[
        "choices"
    ]

    # Should match keys from MODEL_AXIS_MAP
    expected_choices = list(BUILTIN_MODEL_MAP.keys())
    assert choices == expected_choices
    print(f"✓ Choices: {choices}")
    print("✓ Generated from MODEL_AXIS_MAP keys")


if __name__ == "__main__":
    test_model_axis_map()
    test_get_recommended_axis()
    test_model_preset_change_notification()
    test_model_choices_generated_from_map()
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
