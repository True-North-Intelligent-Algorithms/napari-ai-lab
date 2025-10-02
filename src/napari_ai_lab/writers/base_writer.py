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
    def save_labels(
        self, labels_data: np.ndarray, image_path: str, parent_directory: str
    ) -> bool:
        """
        Save labels data to storage.

        Args:
            labels_data: Label array to save
            image_path: Path to the image file these labels belong to
            parent_directory: Parent directory containing the image

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def load_labels(
        self,
        image_path: str,
        parent_directory: str,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """
        Load labels data from storage.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image
            image_shape: Expected shape of the labels array

        Returns:
            Labels array (empty if no saved labels exist)
        """

    @abstractmethod
    def get_labels_path(
        self, image_path: str, parent_directory: str
    ) -> Path | None:
        """
        Get the storage path for labels corresponding to an image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            Path where labels are/will be stored, or None if invalid
        """

    @abstractmethod
    def labels_exist(self, image_path: str, parent_directory: str) -> bool:
        """
        Check if saved labels exist for the given image.

        Args:
            image_path: Path to the image file
            parent_directory: Parent directory containing the image

        Returns:
            True if labels exist, False otherwise
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

    def get_base_labels_directory(self, parent_directory: str) -> Path:
        """
        Get the base directory for storing labels.

        Args:
            parent_directory: Parent directory containing images

        Returns:
            Path to annotations/subdirectory/ folder
        """
        return Path(parent_directory) / "annotations" / self.subdirectory
