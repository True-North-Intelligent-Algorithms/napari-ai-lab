"""
Test TIFF file loading functionality.
"""

from pathlib import Path

import pytest


def test_load_tiff_file():
    """Test loading a TIFF file and inspecting its shape."""
    # Path to test TIFF file
    test_image_path = (
        Path(__file__).parent / "test_images" / "tif" / "TestHidden_002.tif"
    )

    # Check if file exists
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    # Load TIFF file
    try:
        import tifffile

        image_data = tifffile.imread(str(test_image_path))

        # Print information
        print(f"\nTIFF file: {test_image_path.name}")
        print(f"Shape: {image_data.shape}")
        print(f"Data type: {image_data.dtype}")
        print(f"Size (bytes): {image_data.nbytes}")

        # Basic assertions
        assert image_data is not None, "Image data should not be None"
        assert len(image_data.shape) > 0, "Image should have dimensions"
        assert image_data.size > 0, "Image should have non-zero size"

    except ImportError:
        pytest.skip("tifffile package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to load TIFF file: {e}")
