"""
Numpy I/O for label storage.

This module implements label storage using numpy .npy files.
"""

from pathlib import Path

import numpy as np

from .base_io import BaseIO


class NumpyIO(BaseIO):
    """
    I/O implementation that stores labels as numpy .npy files.

    Labels are stored in: parent_directory/annotations/{subdirectory}/{image_name}.npy
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the numpy I/O.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)

    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
    ) -> bool:
        """
        Save data as a numpy .npy file.
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
