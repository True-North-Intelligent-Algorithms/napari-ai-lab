"""
Base writer abstract class for label stora    def load_labels(
        self,
        image_path: str,
        parent_directory: str,
        image_shape: tuple[int, ...]
    ) -> np.ndarray:
This module defines the common interface that all label writers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseWriter(ABC):
    """
    Abstract base class for label writers.

    All label writers must implement this interface to provide consistent
    save/load functionality across different storage formats.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the writer.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        self.subdirectory = subdirectory

    @abstractmethod
    def save(
        self, save_directory: str, image_name: str, data: np.ndarray
    ) -> bool:
        """
        Save data to storage.

        Args:
            save_directory: Directory where data should be saved
            image_name: Name of the image (without extension)
            data: Array to save

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def load(self, load_directory: str, image_name: str) -> np.ndarray:
        """
        Load data from storage.

        Args:
            load_directory: Directory where data is stored
            image_name: Name of the image (without extension)

        Returns:
            Array data (empty if no saved data exists)
        """

    def ensure_labels_directory(self, labels_path: Path) -> None:
        """
        Ensure the directory for labels storage exists.

        Args:
            labels_path: Path where labels will be stored
        """
        if labels_path:
            labels_path.parent.mkdir(parents=True, exist_ok=True)

    def get_image_name(self, image_path: str) -> str:
        """
        Get the base name of an image (without extension).

        Args:
            image_path: Path to the image file

        Returns:
            Base name without extension
        """
        return Path(image_path).stem
