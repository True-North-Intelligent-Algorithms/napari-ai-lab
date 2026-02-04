"""
TIFF I/O for label storage.

This module implements label storage using TIFF files.
"""

from pathlib import Path

import numpy as np
import tifffile

from .base_io import BaseIO


class TiffIO(BaseIO):
    """
    I/O implementation that stores labels as TIFF files.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the TIFF I/O.
        """
        super().__init__(subdirectory)

    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
    ) -> bool:
        try:
            path = Path(save_directory) / f"{dataset_name}.tif"
            tifffile.imwrite(str(path), data)
            return True
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error saving: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        try:
            path = Path(load_directory) / f"{dataset_name}.tif"
            if path.exists():
                return tifffile.imread(str(path)).astype(np.uint16)
            else:
                return np.array([])
        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error loading: {e}")
            return np.array([])
