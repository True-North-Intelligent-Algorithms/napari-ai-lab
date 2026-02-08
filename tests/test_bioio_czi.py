"""
Test CZI file loading with bioio library.
"""

from pathlib import Path

import pytest


@pytest.mark.bioio
def test_load_czi_with_bioio():
    """Test loading a CZI file using BioImage and inspecting its shape and dimensions."""
    # Path to test CZI file
    test_image_path = (
        Path(__file__).parent
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    # Check if file exists
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from bioio import BioImage
    except ImportError:
        pytest.skip("bioio package not installed")

    try:
        # Load image with automatic metadata extraction
        img = BioImage(str(test_image_path))

        # Print information
        print(f"\nCZI file: {test_image_path.name}")
        print(f"Shape: {img.shape}")
        print(f"Dimension order: {img.dims.order}")
        print(f"Data type: {img.data.dtype}")

        # Basic assertions
        assert img.data is not None, "Image data should not be None"
        assert len(img.shape) > 0, "Image should have dimensions"
        assert (
            img.dims.order is not None
        ), "Dimension order should be available"

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to load CZI file with bioio: {e}")
