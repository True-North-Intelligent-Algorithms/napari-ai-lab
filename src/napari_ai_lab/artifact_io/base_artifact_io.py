"""
Base artifact I/O abstract class for storing artifacts associated with image data.

This module defines the common interface that all artifact I/O implementations must implement.
Artifacts include labels, predictions, embeddings, and other data products generated from
or associated with image datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseArtifactIO(ABC):
    """
    Abstract base class for artifact I/O (read/write).

    This class provides a unified interface for reading and writing artifacts associated
    with image data. Artifacts can include labels, predictions, embeddings, features,
    and other derived data products.

    All concrete artifact I/O implementations must implement this interface to provide
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
