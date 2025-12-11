"""
Simple derived model with explicit local paths.
"""

from pathlib import Path

from .image_data_model import ImageDataModel


class LocalImageDataModel(ImageDataModel):
    """
    Derived model that uses explicit local directories for labels and predictions.
    """

    def __init__(
        self, parent_directory: str, labels_dir: str, predictions_dir: str
    ):
        """
        Initialize with explicit directories for labels and predictions.

        Args:
            parent_directory: Path to directory containing images
            labels_dir: Path to directory for storing labels
            predictions_dir: Path to directory for storing predictions
        """
        super().__init__(parent_directory)
        self.labels_dir = Path(labels_dir)
        self.predictions_dir = Path(predictions_dir)

    def get_annotations_directory(self, subdirectory: str = "class_0") -> Path:
        """
        Return explicit local directory for annotations.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")

        Returns:
            Path to configured labels directory
        """
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        return self.labels_dir

    def get_predictions_directory(self, algorithm: str | None = None) -> Path:
        """
        Return explicit local directory for predictions.

        Args:
            algorithm: Optional algorithm name (not used)

        Returns:
            Path to configured predictions directory
        """
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        return self.predictions_dir
