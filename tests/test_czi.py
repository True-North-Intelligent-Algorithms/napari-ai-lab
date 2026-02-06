"""
Test CZI file loading functionality.
"""

from pathlib import Path

import pytest


def test_load_czi_file():
    """Test loading a CZI file and inspecting its shape and axes."""
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

    # Load CZI file
    try:
        from czifile import CziFile

        with CziFile(str(test_image_path)) as czi:
            image_data = czi.asarray()
            axes = czi.axes

            # Print information
            print(f"\nCZI file: {test_image_path.name}")
            print(f"Shape: {image_data.shape}")
            print(f"Axes: {axes}")
            print(f"Data type: {image_data.dtype}")

            # Basic assertions
            assert image_data is not None, "Image data should not be None"
            assert len(image_data.shape) > 0, "Image should have dimensions"
            assert axes is not None, "Axes information should be available"
            assert len(axes) == len(
                image_data.shape
            ), "Axes length should match shape dimensions"

    except ImportError:
        pytest.skip("czifile package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to load CZI file: {e}")


def test_czi_axes_parsing():
    """Test that CZI axes can be parsed correctly."""
    test_image_path = (
        Path(__file__).parent
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    try:
        from czifile import CziFile

        with CziFile(str(test_image_path)) as czi:
            axes = czi.axes

            # Check that axes is a string
            assert isinstance(axes, str), "Axes should be a string"

            # Check for common CZI axes characters
            valid_axes = set("TCZYX0HSRBV")
            for axis in axes:
                assert (
                    axis in valid_axes
                ), f"Axis '{axis}' not in valid CZI axes: {valid_axes}"

            print(f"\nValid axes found: {axes}")

    except ImportError:
        pytest.skip("czifile package not installed")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to parse CZI axes: {e}")


def test_save_czi_channels_to_tiff():
    """Test extracting middle timepoint for each channel and saving as TIFF files."""
    import numpy as np

    test_image_path = (
        Path(__file__).parent
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    if not test_image_path.exists():
        pytest.skip(f"Test image not found: {test_image_path}")

    # Create output directory
    output_dir = Path(__file__).parent / "test_images" / "tif"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tifffile
        from czifile import CziFile

        with CziFile(str(test_image_path)) as czi:
            image_data = czi.asarray()
            axes = czi.axes

            print(f"\nProcessing CZI file: {test_image_path.name}")
            print(f"Shape: {image_data.shape}")
            print(f"Axes: {axes}")

            # Find T (time) and C (channel) axes
            if "T" not in axes or "C" not in axes:
                pytest.skip("CZI file does not have both T and C axes")

            t_idx = axes.index("T")
            c_idx = axes.index("C")

            num_timepoints = image_data.shape[t_idx]
            num_channels = image_data.shape[c_idx]

            # Get middle timepoint
            mid_timepoint = num_timepoints // 2

            print(f"Number of timepoints: {num_timepoints}")
            print(f"Number of channels: {num_channels}")
            print(f"Using middle timepoint: {mid_timepoint}")

            # Save each channel at middle timepoint
            for channel in range(num_channels):
                # Build slice to extract this channel at middle timepoint
                slices = [slice(None)] * len(axes)
                slices[t_idx] = mid_timepoint
                slices[c_idx] = channel

                channel_data = image_data[tuple(slices)]

                # Remove singleton dimensions
                channel_data = np.squeeze(channel_data)

                # Save as TIFF
                output_path = (
                    output_dir
                    / f"{test_image_path.stem}_t{mid_timepoint}_c{channel}.tif"
                )
                tifffile.imwrite(str(output_path), channel_data)

                print(
                    f"Saved channel {channel} (shape: {channel_data.shape}) to {output_path.name}"
                )

                # Verify file was created
                assert (
                    output_path.exists()
                ), f"Output file not created: {output_path}"

            print(f"\nSuccessfully saved {num_channels} channel TIFF files")

    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to save CZI channels to TIFF: {e}")
