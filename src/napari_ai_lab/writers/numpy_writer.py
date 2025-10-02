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

    def save_labels(
        self, labels_data: np.ndarray, image_path: str, parent_directory: str
    ) -> bool:
        """
        Save labels data as a numpy .npy file.

        Args:
            labels_data: Label array to save
            image_path: Path to the image file these labels belong to
            parent_directory: Parent directory containing the image

        Returns:
            True if successful, False otherwise
        """
        print("NumpyWriter: Saving labels for current context:")
        print(f"  Image path: {image_path}")
        print(f"  Parent directory: {parent_directory}")
        print(f"  Labels shape: {labels_data.shape}")

        try:
            labels_path = self.get_labels_path(image_path, parent_directory)

            if not labels_path:
                print("✗ Cannot save - invalid path")
                return False

            print(f"  Target save path: {labels_path}")

            # Ensure directory exists
            self.ensure_labels_directory(labels_path)

            # Convert to uint16 and get stats
            labels_data = labels_data.astype(np.uint16)
            unique_labels = np.unique(labels_data)

            print(f"  Unique labels being saved: {unique_labels}")

            # Save as numpy file
            np.save(str(labels_path), labels_data)
            print(f"✓ Successfully saved labels to: {labels_path}")
            return True

        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error saving labels: {e}")
            return False

    def load_labels(
        self,
        image_path: str,
        parent_directory: str,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """
        Load labels data from numpy .npy file.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image
            image_shape: Expected shape of the labels array

        Returns:
            Labels array (empty if no saved labels exist)
        """
        print("NumpyWriter: Loading labels for image context:")
        print(f"  Image path: {image_path}")
        print(f"  Parent directory: {parent_directory}")
        print(f"  Target image shape: {image_shape}")

        try:
            labels_path = self.get_labels_path(image_path, parent_directory)

            if not labels_path:
                print("ERROR: Cannot load - invalid path")
                return np.zeros(image_shape, dtype=np.uint16)

            print(f"  Expected numpy path: {labels_path}")

            if labels_path.exists():
                print(f"✓ Loading existing labels from: {labels_path}")
                labels_data = np.load(str(labels_path))

                # Ensure the loaded labels match the image shape
                if labels_data.shape == image_shape:
                    print(
                        f"✓ Loaded labels shape matches: {labels_data.shape}"
                    )
                    unique_labels = np.unique(labels_data)
                    print(f"✓ Unique labels in file: {unique_labels}")
                    return labels_data.astype(np.uint16)
                else:
                    print(
                        f"✗ Label shape mismatch: {labels_data.shape} != {image_shape}"
                    )
                    print("Creating new empty labels")
                    return np.zeros(image_shape, dtype=np.uint16)
            else:
                print(f"○ No existing labels found at: {labels_path}")
                print("Creating new empty labels")
                return np.zeros(image_shape, dtype=np.uint16)

        except (OSError, ValueError, PermissionError) as e:
            print(f"✗ Error loading labels: {e}")
            return np.zeros(image_shape, dtype=np.uint16)

    def get_labels_path(
        self, image_path: str, parent_directory: str
    ) -> Path | None:
        """
        Get the numpy file path for labels corresponding to an image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            Path where labels are/will be stored as .npy file, or None if invalid
        """
        if not image_path or not parent_directory:
            return None

        image_name = self.get_image_name(image_path)
        labels_dir = self.get_base_labels_directory(parent_directory)
        return labels_dir / f"{image_name}.npy"

    def labels_exist(self, image_path: str, parent_directory: str) -> bool:
        """
        Check if saved numpy labels exist for the given image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            True if .npy labels file exists, False otherwise
        """
        labels_path = self.get_labels_path(image_path, parent_directory)
        return labels_path is not None and labels_path.exists()
