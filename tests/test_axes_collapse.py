"""
Test axis collapsing for annotation and prediction shape computation.

Simple tests to verify that axes_to_collapse parameter works correctly
for both annotations and predictions.
"""

from napari_ai_lab.utilities.image_util import compute_collapsed_shape


def test_compute_annotation_shape_no_collapse():
    """Test that None or empty axes_to_collapse returns original shape."""
    # No collapse - should match input
    shape = compute_collapsed_shape(
        (10, 512, 512, 3), "ZYXC", axes_to_collapse=None
    )
    assert shape == (10, 512, 512, 3)


def test_compute_annotation_shape_collapse_c():
    """Test collapsing C axis from ZYXC -> ZYX."""
    # Collapse C
    shape = compute_collapsed_shape(
        (10, 512, 512, 3), "ZYXC", axes_to_collapse="C"
    )
    assert shape == (10, 512, 512), f"Expected (10, 512, 512), got {shape}"


def test_compute_annotation_shape_collapse_multiple():
    """Test collapsing multiple axes."""
    # Collapse T and C, keep ZYX
    shape = compute_collapsed_shape(
        (5, 10, 512, 512, 3), "TZYXC", axes_to_collapse=["T", "C"]
    )
    assert shape == (10, 512, 512), f"Expected (10, 512, 512), got {shape}"


def test_compute_annotation_shape_collapse_string_list():
    """Test that both string and list syntax work."""
    # String syntax
    shape1 = compute_collapsed_shape(
        (10, 512, 512, 3), "ZYXC", axes_to_collapse="C"
    )

    # List syntax
    shape2 = compute_collapsed_shape(
        (10, 512, 512, 3), "ZYXC", axes_to_collapse=["C"]
    )

    assert shape1 == shape2 == (10, 512, 512)


def test_compute_annotation_shape_no_axis_types():
    """Test fallback when axis_types not set."""
    # Should return original shape if no axis info
    shape = compute_collapsed_shape(
        (10, 512, 512, 3), None, axes_to_collapse="C"
    )
    assert shape == (10, 512, 512, 3)


def test_same_logic_for_annotations_and_predictions():
    """Test that annotations and predictions use same shape logic."""
    image_shape = (10, 512, 512, 3)

    # Both should compute to same shape with same axes_to_collapse
    ann_shape = compute_collapsed_shape(
        image_shape, "ZYXC", axes_to_collapse="C"
    )
    pred_shape = compute_collapsed_shape(
        image_shape, "ZYXC", axes_to_collapse="C"
    )

    assert ann_shape == pred_shape == (10, 512, 512)


if __name__ == "__main__":
    # Run simple tests
    test_compute_annotation_shape_no_collapse()
    print("✓ No collapse test passed")

    test_compute_annotation_shape_collapse_c()
    print("✓ Collapse C test passed")

    test_compute_annotation_shape_collapse_multiple()
    print("✓ Collapse multiple axes test passed")

    test_compute_annotation_shape_collapse_string_list()
    print("✓ String/list syntax test passed")

    test_compute_annotation_shape_no_axis_types()
    print("✓ No axis types fallback test passed")

    test_same_logic_for_annotations_and_predictions()
    print("✓ Annotations/predictions consistency test passed")

    print("\n✅ All tests passed!")
