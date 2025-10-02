"""
Example TIFF writer for label storage (future implementation).

This module shows how easy it would be to add TIFF support in the future.
Currently just a placeholder/template.
"""

from pathlib import Path

import numpy as np

# from skimage import io  # Would be used for TIFF I/O
from .base_writer import BaseWriter


class TiffWriter(BaseWriter):
    """
    Writer that stores labels as TIFF files (FUTURE IMPLEMENTATION).

    Labels would be stored in: parent_directory/annotations/{subdirectory}/{image_name}.tif
    This makes them easily viewable in ImageJ or other image viewers.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the TIFF writer.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)

    def save_labels(
        self, labels_data: np.ndarray, image_path: str, parent_directory: str
    ) -> bool:
        """
        Save labels data as a TIFF file.

        Args:
            labels_data: Label array to save
            image_path: Path to the image file these labels belong to
            parent_directory: Parent directory containing the image

        Returns:
            True if successful, False otherwise
        """
        # FUTURE IMPLEMENTATION:
        # labels_path = self.get_labels_path(image_path, parent_directory)
        # self.ensure_labels_directory(labels_path)
        # io.imsave(str(labels_path), labels_data.astype(np.uint16))

        print("TiffWriter: TIFF saving not yet implemented")
        return False

    def load_labels(
        self,
        image_path: str,
        parent_directory: str,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """
        Load labels data from TIFF file.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image
            image_shape: Expected shape of the labels array

        Returns:
            Labels array (empty if no saved labels exist)
        """
        # FUTURE IMPLEMENTATION:
        # labels_path = self.get_labels_path(image_path, parent_directory)
        # if labels_path and labels_path.exists():
        #     return io.imread(str(labels_path)).astype(np.uint16)

        print("TiffWriter: TIFF loading not yet implemented")
        return np.zeros(image_shape, dtype=np.uint16)

    def get_labels_path(
        self, image_path: str, parent_directory: str
    ) -> Path | None:
        """
        Get the TIFF file path for labels corresponding to an image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            Path where labels are/will be stored as .tif file, or None if invalid
        """
        if not image_path or not parent_directory:
            return None

        image_name = self.get_image_name(image_path)
        labels_dir = self.get_base_labels_directory(parent_directory)
        return labels_dir / f"{image_name}.tif"

    def labels_exist(self, image_path: str, parent_directory: str) -> bool:
        """
        Check if saved TIFF labels exist for the given image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            True if .tif labels file exists, False otherwise
        """
        labels_path = self.get_labels_path(image_path, parent_directory)
        return labels_path is not None and labels_path.exists()
