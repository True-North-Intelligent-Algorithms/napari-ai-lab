"""
Test TIFF file loading with ndevio library.
"""

from pathlib import Path

import pytest


def test_load_tiff_with_ndevio():
    """Test loading a TIFF file using ndevio nImage for metadata extraction."""
    # Path to test TIFF file
    test_image_path = (
        Path(__file__).parent / "test_images" / "tif" / "TestHidden_002.tif"
    )

    # Check if file exists
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from ndevio import nImage

        # Load image with automatic metadata extraction
        img = nImage(str(test_image_path))

        # Print nImage information
        print(f"\nTIFF file loaded with ndevio: {test_image_path.name}")
        print(f"Dimensions: {img.dims}")
        print(f"Layer data shape: {img.layer_data.shape}")
        print(f"Layer data dtype: {img.layer_data.dtype}")

        # Basic assertions
        assert img is not None, "nImage object should not be None"
        assert img.dims is not None, "Dimensions should be available"
        assert img.layer_data is not None, "Layer data should be available"
        assert (
            len(img.layer_data.shape) > 0
        ), "Layer data should have dimensions"

        # Print additional metadata if available
        if hasattr(img, "metadata"):
            print(f"Metadata: {img.metadata}")

    except ImportError:
        pytest.skip("ndevio package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to load TIFF file with ndevio: {e}")


def test_parse_dimensions_with_ndevio():
    """Test parsing image dimensions from a TIFF file using ndevio."""
    # Path to test TIFF file
    test_image_path = (
        Path(__file__).parent / "test_images" / "tif" / "TestHidden_002.tif"
    )

    # Check if file exists
    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from ndevio import nImage

        # Load image with automatic metadata extraction
        img = nImage(str(test_image_path))

        # Access and print image dimensions
        dims = img.dims
        print(f"\nImage dimensions: {dims}")

        # Basic assertion
        assert dims is not None, "Dimensions should be available"

    except ImportError:
        pytest.skip("ndevio package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to parse dimensions with ndevio: {e}")
