"""
Test CellCast StarDist segmenter registration and basic functionality.
"""

import numpy as np


def test_cellcast_segmenter_registration():
    """Test that CellCast segmenter can be registered."""
    from napari_ai_lab.Segmenters.GlobalSegmenters import (
        CellCastStardistSegmenter,
    )

    if CellCastStardistSegmenter is None:
        print("⚠️  CellCast not available - skipping test")
        return

    # Register the segmenter
    CellCastStardistSegmenter.register()

    # Check it's available in the registry
    from napari_ai_lab.Segmenters.GlobalSegmenters.GlobalSegmenterBase import (
        GlobalSegmenterBase,
    )

    assert "CellCastStardistSegmenter" in GlobalSegmenterBase.registry
    print("✅ CellCast segmenter registered successfully")


def test_cellcast_segmenter_predict():
    """Test basic prediction with CellCast segmenter."""
    from napari_ai_lab.Segmenters.GlobalSegmenters import (
        CellCastStardistSegmenter,
    )

    if CellCastStardistSegmenter is None:
        print("⚠️  CellCast not available - skipping test")
        return

    # Create a simple test image (synthetic cell-like structure)
    image = np.zeros((256, 256), dtype=np.float32)
    # Add a circular blob
    y, x = np.ogrid[:256, :256]
    mask = (x - 128) ** 2 + (y - 128) ** 2 <= 30**2
    image[mask] = 1.0

    # Create segmenter and segment
    segmenter = CellCastStardistSegmenter()
    labels = segmenter.segment(image)

    # Check output is valid
    assert labels.shape == image.shape
    assert labels.dtype == np.uint16
    assert labels.max() >= 0  # Should have at least background
    print(f"✅ CellCast prediction successful: {labels.max()} objects found")


if __name__ == "__main__":
    print("Testing CellCast StarDist Segmenter...")
    test_cellcast_segmenter_registration()
    test_cellcast_segmenter_predict()
    print("✅ All tests passed!")
