"""
Numpy writer for label storage.

This module implements label storage using numpy .npy files.
"""

from pathlib import Path

import numpy as np

from .base_writer import BaseWriter


class NumpyWriter(BaseWriter):
    """
    Writer that stores labels as numpy .npy files.

    Labels are stored in: parent_directory/annotations/{subdirectory}/{image_name}.npy
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the numpy writer.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)

    def save(
        self, save_directory: str, dataset_name: str, data: np.ndarray
    ) -> bool:
        """
        Save data as a numpy .npy file.

        Args:
            save_directory: Directory where data should be saved
            dataset_name: Name of the dataset (without extension)
            data: Array to save

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(save_directory) / f"{dataset_name}.npy"
            np.save(str(path), data)
            return True
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error saving: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        """
        Load data from numpy .npy file.

        Args:
            load_directory: Directory where data is stored
            dataset_name: Name of the dataset (without extension)

        Returns:
            Array data (empty if no saved data exists)
        """
        try:
            path = Path(load_directory) / f"{dataset_name}.npy"
            if path.exists():
                return np.load(str(path))
            else:
                return np.array([])
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error loading: {e}")
            return np.array([])
