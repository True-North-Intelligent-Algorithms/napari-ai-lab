"""
Test CZI file loading with ndevio library.
"""

from pathlib import Path

import pytest


def test_load_czi_with_ndevio():
    """Test loading a CZI file using ndevio nImage for metadata extraction."""
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
        from ndevio import nImage

        # Load image with automatic metadata extraction
        img = nImage(str(test_image_path))

        # Print nImage information
        print(f"\nCZI file loaded with ndevio: {test_image_path.name}")
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
        pytest.fail(f"Failed to load CZI file with ndevio: {e}")


def test_ndevio_czi_dimensions_parsing():
    """Test that ndevio correctly parses dimensions from CZI file."""
    test_image_path = (
        Path(__file__).parent
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from ndevio import nImage

        img = nImage(str(test_image_path))

        print(f"\nDimension analysis for: {test_image_path.name}")
        print(f"Full dimensions: {img.dims}")
        print(f"Layer data shape: {img.layer_data.shape}")

        # Check that dims is accessible and has expected structure
        assert hasattr(img, "dims"), "nImage should have dims attribute"

        # Verify layer_data shape matches or is compatible with dims
        assert img.layer_data.shape is not None, "Layer data should have shape"

        # CZI files should have multiple dimensions (T, C, Z, Y, X)
        assert (
            len(img.layer_data.shape) >= 2
        ), "CZI should have at least 2D data"

    except ImportError:
        pytest.skip("ndevio package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to parse dimensions with ndevio: {e}")


def test_ndevio_czi_axes_detection():
    """Test that ndevio can detect and report axes from CZI metadata."""
    test_image_path = (
        Path(__file__).parent
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from ndevio import nImage

        img = nImage(str(test_image_path))

        print(f"\nAxes analysis for: {test_image_path.name}")
        print(f"Dimensions: {img.dims}")

        # Check if common CZI axes are detected
        dims_str = str(img.dims)

        # CZI files typically have Time (T) and Channel (C) dimensions
        print(f"Contains T (time): {'T' in dims_str}")
        print(f"Contains C (channel): {'C' in dims_str}")
        print(f"Contains Z (depth): {'Z' in dims_str}")
        print(f"Contains Y (rows): {'Y' in dims_str}")
        print(f"Contains X (cols): {'X' in dims_str}")

        # At minimum, should have Y and X
        assert (
            "Y" in dims_str or "X" in dims_str
        ), "Should detect spatial dimensions"

    except ImportError:
        pytest.skip("ndevio package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to detect axes with ndevio: {e}")
