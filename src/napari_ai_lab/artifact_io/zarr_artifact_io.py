"""
Zarr artifact I/O for storing artifacts.

This module implements artifact storage using Zarr format, which is efficient
for large multi-dimensional arrays and supports compression and smart slicing.
"""

from pathlib import Path

import numpy as np
import zarr

from .base_artifact_io import BaseArtifactIO


class ZarrArtifactIO(BaseArtifactIO):
    """
    I/O implementation that stores labels as Zarr arrays with smart slicing.

    Zarr is particularly useful for large datasets as it supports:
    - Compression
    - Chunked storage
    - Efficient partial reads/writes
    - Smart slicing: save individual slices into a full zarr array
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the Zarr I/O.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)
        self.shape_total = None  # Total size of the full array
        self.axis_slice = None  # Axis string for slicing (e.g., "YX", "ZYX")

    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
        selected_axis: str = None,
    ) -> bool:
        """
        Save data as a Zarr array slice or full array.

        If shape_total and current_step are set, saves only the slice at current_step
        into a larger zarr array. Otherwise saves the full data.

        Args:
            save_directory: Directory where data should be saved
            dataset_name: Name of the dataset (without extension)
            data: Array to save (can be a slice)
            current_step: Viewer dimension position (for stacked mode)
            selected_axis: Axis string like "YX", "ZYX", "YXC", etc.

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(save_directory) / f"{dataset_name}.zarr"

            # Smart slicing mode: if shape_total and current_step are set
            if (
                self.shape_total is not None
                and current_step is not None
                and self.axis_slice is not None
            ):
                # Determine spatial dimensions from axis_slice
                if self.axis_slice.endswith("ZYX"):
                    spatial_dims = 3
                elif self.axis_slice.endswith("YX"):
                    spatial_dims = 2
                else:
                    spatial_dims = 2

                # Open or create zarr array with the total shape
                if path.exists():
                    # Open existing zarr array
                    z = zarr.open(str(path), mode="r+")
                else:
                    # Create new zarr array
                    z = zarr.open(
                        str(path),
                        mode="w",
                        shape=self.shape_total,
                        chunks=True,
                        dtype=np.uint16,
                    )

                # Extract non-spatial indices from current_step
                non_spatial_indices = current_step[:-spatial_dims]

                # Build the slice index for the zarr array
                slice_index = non_spatial_indices + tuple(
                    [slice(None)] * spatial_dims
                )

                # Write the data slice into the zarr array
                z[slice_index] = data

            else:
                # Simple mode: save entire array
                zarr.save(str(path), data)

            return True
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error saving: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        """
        Load data from Zarr array.

        Args:
            load_directory: Directory where data is stored
            dataset_name: Name of the dataset (without extension)

        Returns:
            Array data (empty if no saved data exists)
        """
        try:
            path = Path(load_directory) / f"{dataset_name}.zarr"
            if path.exists():
                # Load zarr array and convert to numpy array
                data = zarr.load(str(path))
                return np.asarray(data, dtype=np.uint16)
            else:
                return np.array([])
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error loading: {e}")
            return np.array([])

    def set_shape_total(self, shape_total: tuple):
        """
        Set the total shape of the full array.

        Args:
            shape_total: Total shape of the full array
        """
        self.shape_total = shape_total

    def set_axis_slice(self, axis_slice: str):
        """
        Set the axis slice string (e.g., "YX", "ZYX", "TZYX").

        Args:
            axis_slice: Axis string describing the dimensions (e.g., "YX", "ZYX", "TZYX")
        """
        self.axis_slice = axis_slice
