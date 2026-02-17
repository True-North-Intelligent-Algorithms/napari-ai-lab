"""Test for TiffSliceIO artifact storage."""

import shutil
from pathlib import Path

import numpy as np

from napari_ai_lab.artifact_io import TiffSliceIO


def test_tiff_slice_io_3d_sparse_save_load():
    """
    Test TiffSliceIO with 3D data that has values only at two slices.

    Creates 128x128x16 volume with data only at slices 5 and 11.
    Saves using XY mode (should save only 2 files for the 2 non-zero slices).
    Loads back and verifies data matches.
    """
    # Create test directory under test_images
    test_base_dir = Path(__file__).parent / "test_images" / "test_tiff_slice"

    # Clean up if it exists from previous run
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)

    # Create the test directory
    test_base_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create 3D data: 16 slices of 128x128
        shape_total = (16, 128, 128)  # ZYX format
        data_3d = np.zeros(shape_total, dtype=np.uint16)

        # Put data only at slice 5 and slice 11
        data_3d[5, :, :] = 100  # Fill slice 5 with value 100
        data_3d[11, :, :] = 200  # Fill slice 11 with value 200

        # Create a pattern to make it easier to verify
        data_3d[5, 10:20, 10:20] = 150  # Small square in slice 5
        data_3d[11, 30:40, 30:40] = 250  # Small square in slice 11

        save_dir = test_base_dir / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create TiffSliceIO instance
        io = TiffSliceIO(subdirectory="predictions")

        # Set up the slice parameters
        # Only need to set shape_total and axis_slice
        # The slice shape is derived from these (last 2 dims for YX, last 3 for ZYX)
        io.set_shape_total(shape_total)
        io.set_axis_slice("YX")

        dataset_name = "test_volume"

        # Save each slice that has data
        # Slice 5
        current_step_5 = (5, 0, 0)  # Z=5, Y=0, X=0
        success_5 = io.save(
            str(save_dir),
            dataset_name,
            data_3d[5, :, :],  # Just the 2D slice
            current_step=current_step_5,
            selected_axis="YX",
        )

        # Slice 11
        current_step_11 = (11, 0, 0)  # Z=11, Y=0, X=0
        success_11 = io.save(
            str(save_dir),
            dataset_name,
            data_3d[11, :, :],  # Just the 2D slice
            current_step=current_step_11,
            selected_axis="YX",
        )

        assert success_5, "Failed to save slice 5"
        assert success_11, "Failed to save slice 11"

        # Verify only 2 files were saved
        saved_files = list(save_dir.glob("*.tif"))
        assert (
            len(saved_files) == 2
        ), f"Expected 2 files, found {len(saved_files)}"

        # Check the filenames
        expected_files = {
            save_dir / "test_volume_5.tif",
            save_dir / "test_volume_11.tif",
        }
        actual_files = set(saved_files)
        assert (
            actual_files == expected_files
        ), f"File names don't match. Expected {expected_files}, got {actual_files}"

        # Now load back the full volume
        loaded_data = io.load(str(save_dir), dataset_name)

        # Verify shape
        assert (
            loaded_data.shape == shape_total
        ), f"Shape mismatch: expected {shape_total}, got {loaded_data.shape}"

        # Verify slice 5 data
        assert np.array_equal(
            loaded_data[5, :, :], data_3d[5, :, :]
        ), "Slice 5 data doesn't match"

        # Verify slice 11 data
        assert np.array_equal(
            loaded_data[11, :, :], data_3d[11, :, :]
        ), "Slice 11 data doesn't match"

        # Verify other slices are zeros
        for z in range(16):
            if z not in [5, 11]:
                assert np.all(
                    loaded_data[z, :, :] == 0
                ), f"Slice {z} should be all zeros"

        # Verify specific patterns
        assert np.all(
            loaded_data[5, 10:20, 10:20] == 150
        ), "Pattern in slice 5 doesn't match"
        assert np.all(
            loaded_data[11, 30:40, 30:40] == 250
        ), "Pattern in slice 11 doesn't match"

        print(
            "✓ Test passed: TiffSliceIO correctly saved 2 files and loaded them back"
        )

        # Test passed - clean up the test directory
        shutil.rmtree(test_base_dir)
        print(f"✓ Cleaned up test directory: {test_base_dir}")

    except Exception as e:
        # Test failed - keep directory for inspection
        print(f"✗ Test failed! Test data preserved at: {test_base_dir}")
        print(f"  Error: {e}")
        raise


def test_tiff_slice_io_fallback_single_file():
    """
    Test TiffSliceIO fallback to single file load when shape_total is not set.
    """
    # Create test directory under test_images
    test_base_dir = (
        Path(__file__).parent / "test_images" / "test_tiff_slice_fallback"
    )

    # Clean up if it exists from previous run
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir)

    # Create the test directory
    test_base_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_dir = test_base_dir / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create simple 2D data
        data_2d = np.random.randint(0, 100, size=(64, 64), dtype=np.uint16)

        io = TiffSliceIO(subdirectory="predictions")
        dataset_name = "test_2d"

        # Save without setting shape_total (should just save single file)
        success = io.save(
            str(save_dir),
            dataset_name,
            data_2d,
            current_step=None,
            selected_axis=None,
        )

        assert success, "Failed to save 2D data"

        # Verify single file saved
        saved_files = list(save_dir.glob("*.tif"))
        assert (
            len(saved_files) == 1
        ), f"Expected 1 file, found {len(saved_files)}"

        # Load back (should use fallback single file load)
        loaded_data = io.load(str(save_dir), dataset_name)

        assert np.array_equal(
            loaded_data, data_2d
        ), "2D data doesn't match after load"

        print("✓ Test passed: TiffSliceIO fallback to single file works")

        # Test passed - clean up the test directory
        shutil.rmtree(test_base_dir)
        print(f"✓ Cleaned up test directory: {test_base_dir}")

    except Exception as e:
        # Test failed - keep directory for inspection
        print(f"✗ Test failed! Test data preserved at: {test_base_dir}")
        print(f"  Error: {e}")
        raise


if __name__ == "__main__":
    test_tiff_slice_io_3d_sparse_save_load()
    test_tiff_slice_io_fallback_single_file()
    print("\n✓ All TiffSliceIO tests passed!")
