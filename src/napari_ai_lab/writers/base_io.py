"""
Base I/O abstract class for label storage.

This module defines the common interface that all label I/O implementations must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseIO(ABC):
    """
    Abstract base class for I/O (read/write).

    All concrete I/O implementations must implement this interface to provide
    consistent save/load functionality across different storage formats.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the I/O implementation.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        self.subdirectory = subdirectory

    @abstractmethod
    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
    ) -> bool:
        """
        Save data to storage.

        Args:
            save_directory: Directory where data should be saved
            dataset_name: Name of the dataset (without extension)
            data: Array to save
            current_step: Viewer dimension position (for stacked mode)

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        """
        Load data from storage.

        Args:
            load_directory: Directory where data is stored
            dataset_name: Name of the dataset (without extension)

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

    def get_dataset_name(self, dataset_path: str) -> str:
        """
        Get the base name of a dataset (without extension).

        Args:
            dataset_path: Path to the dataset file

        Returns:
            Base name without extension
        """
        return Path(dataset_path).stem
