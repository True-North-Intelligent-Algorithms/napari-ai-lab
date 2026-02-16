"""
TIFF Slice artifact I/O for storing artifacts as hyperslices.

This module implements artifact storage using TIFF files where large arrays
are saved as individual hyperslice files and reconstructed on load.
"""

from pathlib import Path

import numpy as np
import tifffile

from ..utility import create_artifact_name
from .base_artifact_io import BaseArtifactIO


class TiffSliceIO(BaseArtifactIO):
    """
    I/O implementation that stores artifacts as individual hyperslice TIFF files.

    This class allows saving large multi-dimensional arrays as separate slices
    along a specified axis, and reconstructing them on load.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the TIFF Slice I/O.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)
        self.shape_total = None  # Total size of the full array
        self.shape_slice = None  # Size of each hyperslice
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
        Save data as TIFF file(s), using selected_axis to determine slicing.

        If selected_axis is provided, always use it to create the artifact name.
        If the axis represents the full array (e.g., "all" or matches full dimensions),
        save all slices individually.

        Args:
            save_directory: Directory where data should be saved
            dataset_name: Name of the dataset (without extension)
            data: Array to save
            current_step: Viewer dimension position (for stacked mode)
            selected_axis: Axis string like "YX", "ZYX", "YXC", etc.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Always use selected_axis if provided
            if selected_axis is not None and current_step is not None:
                dataset_name = create_artifact_name(
                    dataset_name, current_step, selected_axis
                )

            path = Path(save_directory) / f"{dataset_name}.tif"
            tifffile.imwrite(str(path), data)
            return True
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error saving: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        """
        Load data by reconstructing from individual hyperslice TIFF files.

        This method uses the stored shape_total, shape_slice, and axis_slice members
        to determine how to loop through all potential hyperslices, load them,
        and reconstruct the full array.

        Args:
            load_directory: Directory where data is stored
            dataset_name: Name of the dataset (without extension)

        Returns:
            Array data (empty if no saved data exists)
        """
        # Check if we need to reconstruct from slices
        if self.shape_total is None or self.axis_slice is None:
            # Fall back to simple single-file load
            return self._load_single_file(load_directory, dataset_name)

        # Determine spatial dimensions from axis_slice
        if self.axis_slice.endswith("ZYX"):
            spatial_dims = 3
        elif self.axis_slice.endswith("YX"):
            spatial_dims = 2
        else:
            spatial_dims = 2

        # Calculate non-spatial dimensions
        non_spatial_shape = self.shape_total[:-spatial_dims]

        # Initialize the full array with zeros
        full_array = np.zeros(self.shape_total, dtype=np.uint16)

        # If there are no non-spatial dimensions, just load the single file
        if len(non_spatial_shape) == 0:
            return self._load_single_file(load_directory, dataset_name)

        # Generate all possible non-spatial indices and load corresponding slices
        indices_ranges = [range(dim_size) for dim_size in non_spatial_shape]

        # Use numpy's ndindex to iterate over all combinations
        import itertools

        for idx in itertools.product(*indices_ranges):
            # Create the step tuple (non-spatial indices + zeros for spatial)
            step = idx + tuple([0] * spatial_dims)

            # Create the hyperslice name
            slice_name = create_artifact_name(
                dataset_name, step, self.axis_slice
            )
            slice_path = Path(load_directory) / f"{slice_name}.tif"

            if slice_path.exists():
                # Load the slice
                slice_data = tifffile.imread(str(slice_path)).astype(np.uint16)

                # Put the slice into the correct position in the full array
                full_slice_index = idx + tuple([slice(None)] * spatial_dims)
                full_array[full_slice_index] = slice_data
            else:
                # No file exists for this slice - it remains zeros
                pass

        return full_array

    def _load_single_file(
        self, load_directory: str, dataset_name: str
    ) -> np.ndarray:
        """
        Helper method to load a single TIFF file.

        Args:
            load_directory: Directory where data is stored
            dataset_name: Name of the dataset (without extension)

        Returns:
            Array data (empty if no saved data exists)
        """
        try:
            path = Path(load_directory) / f"{dataset_name}.tif"
            if path.exists():
                return tifffile.imread(str(path)).astype(np.uint16)
            else:
                return np.array([])
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error loading single file: {e}")
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

        This is a helper method to set the axis_slice member directly.

        Args:
            axis_slice: Axis string describing the dimensions (e.g., "YX", "ZYX", "TZYX")
        """
        self.axis_slice = axis_slice
