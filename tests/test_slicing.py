"""
Test array slicing operations with multi-dimensional image data.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from napari_ai_lab.utility import (
    create_artifact_name,
    get_current_slice_indices,
)


def test_slice_czifile():
    """Test opening a CZI file and printing its shape."""
    start_time = time.time()

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
        from czifile import CziFile
    except ImportError:
        pytest.skip("czifile package not installed")

    try:
        with CziFile(str(test_image_path)) as czi:
            image_data = czi.asarray()
            print(f"Original shape: {image_data.shape}")
            image_data = (
                image_data.squeeze()
            )  # Remove singleton dimensions for better viewing
            axes = czi.axes

            # Print information
            print(f"\nCZI file: {test_image_path.name}")
            print(f"Shape: {image_data.shape}")
            print(f"Axes: {axes}")
            print(f"Data type: {image_data.dtype}")
            data_mean = image_data.mean()
            print(f"Data mean: {data_mean}")

            # Basic assertion
            assert image_data is not None, "Image data should not be None"
            assert len(image_data.shape) > 0, "Image should have dimensions"

            step = (1, 1, 10, 511, 511)

            selected_axis = "ZYX"

            # Slicing the image data
            indices = get_current_slice_indices(step, selected_axis)

            image_name = create_artifact_name(
                test_image_path.name, step, selected_axis
            )
            print(f"Sliced image name: {image_name}")

            data_slice = image_data[indices]
            print(f"Current slice shape: {data_slice.shape}")
            slice_mean = data_slice.mean()
            print(f"Current slice mean: {slice_mean}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Total time (czifile): {elapsed_time:.3f} seconds")

            # Assertions for slice shape and mean
            assert data_slice.shape == (
                44,
                1024,
                1024,
            ), f"Expected shape (44, 1024, 1024), got {data_slice.shape}"

            assert np.isclose(
                slice_mean, 1.42, atol=0.01
            ), f"Expected mean ~1.42, got {slice_mean:.2f}"

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to open CZI file: {e}")


@pytest.mark.bioio
def test_slice_bioio_czi():
    """Test opening a CZI file with bioio and slicing it."""
    start_time = time.time()

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
        img = BioImage(str(test_image_path))
        image_data = img.data
        print(f"Original shape: {image_data.shape}")
        image_data = (
            image_data.squeeze()
        )  # Remove singleton dimensions for better viewing
        axes = img.dims.order

        # Print information
        print(f"\nCZI file: {test_image_path.name}")
        print(f"Shape: {image_data.shape}")
        print(f"Dimension order (axes): {axes}")
        print(f"Data type: {image_data.dtype}")
        data_mean = image_data.mean()
        print(f"Data mean before slicing: {data_mean}")

        # Basic assertion
        assert image_data is not None, "Image data should not be None"
        assert len(image_data.shape) > 0, "Image should have dimensions"

        step = (1, 1, 10, 511, 511)

        selected_axis = "ZYX"

        # Slicing the image data
        indices = get_current_slice_indices(step, selected_axis)

        image_name = create_artifact_name(
            test_image_path.name, step, selected_axis
        )
        print(f"Sliced image name: {image_name}")

        data_slice = image_data[indices]
        print(f"Current slice shape: {data_slice.shape}")
        slice_mean = data_slice.mean()
        print(f"Current slice mean: {slice_mean}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time (bioio): {elapsed_time:.3f} seconds")

        # Assertions for slice shape and mean
        assert data_slice.shape == (
            44,
            1024,
            1024,
        ), f"Expected shape (44, 1024, 1024), got {data_slice.shape}"

        assert np.isclose(
            slice_mean, 1.42, atol=0.01
        ), f"Expected mean ~1.42, got {slice_mean:.2f}"

    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Failed to open CZI file with bioio: {e}")
