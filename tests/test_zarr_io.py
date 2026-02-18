"""Test for ZarrArtifactIO artifact storage."""

import shutil
from pathlib import Path

import numpy as np

from napari_ai_lab.artifact_io import ZarrArtifactIO


def test_zarr_io_3d_sparse_save_load():
    """
    Test ZarrArtifactIO with 3D data that has values only at two slices.

    Creates 128x128x16 volume with data only at slices 5 and 11.
    Saves using Zarr format.
    Loads back and verifies data matches.
    """
    # Create test directory under test_images
    test_base_dir = Path(__file__).parent / "test_images" / "test_zarr"

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

        # Create ZarrArtifactIO instance
        io = ZarrArtifactIO(subdirectory="predictions")

        dataset_name = "test_volume"

        # Save the full 3D volume
        success = io.save(
            str(save_dir),
            dataset_name,
            data_3d,
            current_step=None,
            selected_axis=None,
        )

        assert success, "Failed to save 3D volume"

        # Verify zarr file/directory was created
        zarr_path = save_dir / f"{dataset_name}.zarr"
        assert (
            zarr_path.exists()
        ), f"Zarr file/directory not found at {zarr_path}"

        # Now load back the full volume
        loaded_data = io.load(str(save_dir), dataset_name)
        print("loaded data size", loaded_data.shape)

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
            "✓ Test passed: ZarrArtifactIO correctly saved and loaded 3D data"
        )

        # Test passed - clean up the test directory
        shutil.rmtree(test_base_dir)
        print(f"✓ Cleaned up test directory: {test_base_dir}")

    except Exception as e:
        # Test failed - keep directory for inspection
        print(f"✗ Test failed! Test data preserved at: {test_base_dir}")
        print(f"  Error: {e}")
        raise


def test_zarr_io_fallback_load():
    """
    Test ZarrArtifactIO with simple 2D data.
    """
    # Create test directory under test_images
    test_base_dir = Path(__file__).parent / "test_images" / "test_zarr_2d"

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

        io = ZarrArtifactIO(subdirectory="predictions")
        dataset_name = "test_2d"

        # Save 2D data
        success = io.save(
            str(save_dir),
            dataset_name,
            data_2d,
            current_step=None,
            selected_axis=None,
        )

        assert success, "Failed to save 2D data"

        # Verify zarr file/directory exists
        zarr_path = save_dir / f"{dataset_name}.zarr"
        assert (
            zarr_path.exists()
        ), f"Zarr file/directory not found at {zarr_path}"

        # Load back
        loaded_data = io.load(str(save_dir), dataset_name)

        assert np.array_equal(
            loaded_data, data_2d
        ), "2D data doesn't match after load"

        print("✓ Test passed: ZarrArtifactIO 2D data works")

        # Test passed - clean up the test directory
        shutil.rmtree(test_base_dir)
        print(f"✓ Cleaned up test directory: {test_base_dir}")

    except Exception as e:
        # Test failed - keep directory for inspection
        print(f"✗ Test failed! Test data preserved at: {test_base_dir}")
        print(f"  Error: {e}")
        raise


if __name__ == "__main__":
    test_zarr_io_3d_sparse_save_load()
    test_zarr_io_fallback_load()
    print("\n✓ All ZarrArtifactIO tests passed!")
