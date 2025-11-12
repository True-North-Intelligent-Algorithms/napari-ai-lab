"""
Simple ImageDataModel for managing image directories and file organization.

This model handles:
- Image directory scanning
- Result directory organization
- Path generation for different result types
"""

from pathlib import Path


class ImageDataModel:
    """
    Simple model for managing image data and result organization.

    Handles image directory scanning and provides organized paths
    for saving different types of results (segmentations, embeddings, etc.).
    """

    # Supported image extensions
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, parent_directory: str):
        """
        Initialize with a parent directory containing images.

        Args:
            parent_directory: Path to directory containing images
        """
        self.parent_directory = Path(parent_directory).resolve()
        self.results_directory = self.parent_directory / "results"
        self._image_paths: list[Path] | None = None

        # Load image list on initialization
        self._load_image_list(str(self.parent_directory))

    def get_image_paths(self) -> list[Path]:
        """Get sorted list of image file paths in the directory."""
        if self._image_paths is None:
            self._scan_images()
        return self._image_paths

    def _scan_images(self):
        """Scan directory for supported image files."""
        if not self.parent_directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {self.parent_directory}"
            )

        self._image_paths = [
            f
            for f in self.parent_directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        self._image_paths.sort(key=lambda x: x.name.lower())

    def get_result_path(
        self, result_type: str, algorithm: str, image_path: str
    ) -> Path:
        """
        Get organized path for saving results.

        Args:
            result_type: Type of result ("segmentations", "embeddings", etc.)
            algorithm: Algorithm name ("otsu", "sam3d", etc.)
            image_path: Path to source image

        Returns:
            Path where result should be saved
        """
        # Deprecated: use the organized directories API (get_predictions_directory)
        # This method was removed because callers should use the dedicated
        # directory accessors like get_base_embeddings_directory and
        # get_predictions_directory.
        raise NotImplementedError(
            "get_result_path has been removed; use get_predictions_directory or get_base_embeddings_directory"
        )

    def ensure_result_directories(
        self, result_types: list[str], algorithms: list[str]
    ):
        """
        Create directory structure for results.

        Args:
            result_types: List of result types to create directories for
            algorithms: List of algorithms to create subdirectories for
        """
        for result_type in result_types:
            for algorithm in algorithms:
                result_dir = self.results_directory / result_type / algorithm
                result_dir.mkdir(parents=True, exist_ok=True)

    def get_image_count(self) -> int:
        """Get total number of images in directory."""
        return len(self.get_image_paths())

    def _load_image_list(self, directory: str):
        """
        Load image list from directory - compatible with sequence viewer interface.

        Args:
            directory: Path to directory containing images
        """
        # Update parent directory if different
        new_parent = Path(directory).resolve()
        if new_parent != self.parent_directory:
            self.parent_directory = new_parent
            self.results_directory = self.parent_directory / "results"
            self._image_paths = None

        # Use existing _scan_images functionality
        self._scan_images()

    def get_base_annotations_directory(
        self, subdirectory: str = "class_0"
    ) -> Path:
        """
        Get the base directory for storing annotations.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")

        Returns:
            Path to annotations/subdirectory/ folder
        """
        annotation_dir = self.parent_directory / "annotations" / subdirectory

        annotation_dir.mkdir(parents=True, exist_ok=True)

        return annotation_dir

    def get_base_embeddings_directory(self) -> Path:
        """
        Get the base directory for storing embeddings.

        Returns:
            Path to embeddings/ folder
        """
        return self.parent_directory / "embeddings"

    def get_predictions_directory(self, algorithm: str | None = None) -> Path:
        """
        Get the base directory for storing prediction results.

        Args:
            algorithm: Optional algorithm name (not used yet, reserved for future)

        Returns:
            Path to predictions/ folder (optionally organized by algorithm in future)
        """
        # For now, predictions are stored in a flat `predictions/` folder
        # under the parent directory. The `algorithm` parameter is reserved
        # for future use where we might create subfolders per algorithm.
        return self.parent_directory / "predictions"

    def get_annotations_writer(self):
        """
        Get a label writer for this model.

        Returns:
            BaseWriter: A numpy writer instance for label persistence
        """
        from ..writers import get_writer

        return get_writer("numpy")

    def load_existing_annotations(
        self, image_shape, image_index: int = 0, subdirectory: str = "class_0"
    ):
        """
        Load existing annotation array for the image at image_index, or return an
        empty array matching image_shape if no saved data exists.

        Args:
            image_shape: Shape of the image to match for empty array creation.
            image_index: Index of the image in the model's image list.
            subdirectory: Subdirectory under 'annotations' to look in (default: class_0).

        Returns:
            numpy ndarray containing annotation labels (dtype preserved by writer or uint16 zeros).
        """
        # Defer imports to here to avoid bringing numpy/writer into top-level module load
        import numpy as np

        annotation_dir = self.get_base_annotations_directory(subdirectory)

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            image_name = image_paths[image_index].stem
        else:
            image_name = "unknown"

        writer = self.get_annotations_writer()
        data = writer.load(str(annotation_dir), image_name)

        # If nothing saved, return zeros
        if data is None or getattr(data, "size", 0) == 0:
            return np.zeros(image_shape, dtype=np.uint16)

        return data

    def get_predictions_writer(self):
        """
        Get a writer for prediction outputs (currently same as annotations writer).

        Returns:
            BaseWriter: A numpy writer instance for prediction persistence
        """
        from ..writers import get_writer

        return get_writer("numpy")

    def load_existing_predictions(
        self,
        image_shape,
        image_index: int = 0,
        subdirectory: str = "predictions",
    ):
        """
        Load existing prediction array for the image at image_index, or return an
        empty array matching image_shape if no saved data exists.

        Args:
            image_shape: Shape of the image to match for empty array creation.
            image_index: Index of the image in the model's image list.
            subdirectory: Subdirectory under 'predictions' to look in (default: 'predictions').

        Returns:
            numpy ndarray containing prediction labels (dtype preserved by writer or uint16 zeros).
        """
        import numpy as np

        preds_dir = self.get_predictions_directory(
            subdirectory if subdirectory else "predictions"
        )

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            image_name = image_paths[image_index].stem
        else:
            image_name = "unknown"

        writer = self.get_predictions_writer()
        data = writer.load(str(preds_dir), image_name)

        if data is None or getattr(data, "size", 0) == 0:
            return np.zeros(image_shape, dtype=np.uint16)

        return data

    def save_annotations(
        self, labels_array, image_index: int, subdirectory: str = "class_0"
    ):
        """
        Save the provided labels array for the image at image_index under the
        annotations/subdirectory directory.

        Args:
            labels_array: numpy array containing labels to save.
            image_index: Index of the image to associate the labels with.
            subdirectory: Subdirectory under annotations to save into (default: class_0).

        Returns:
            The result of the writer.save(...) call.
        """
        # Defer heavy imports
        import numpy as np

        annotation_dir = self.get_base_annotations_directory(subdirectory)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            image_name = image_paths[image_index].stem
        else:
            image_name = "unknown"

        writer = self.get_annotations_writer()

        # Ensure uint16 to match previous behavior
        labels_to_save = np.asarray(labels_array).astype(np.uint16)

        return writer.save(str(annotation_dir), image_name, labels_to_save)

    def get_global_frameworks(self):
        """Return the dict of registered global segmenter frameworks, or empty dict."""
        from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase

        return GlobalSegmenterBase.get_registered_frameworks()

    def get_global_framework_names(self) -> list[str]:
        """
        Get the names of all registered global segmenter frameworks.
        """

        frameworks = self.get_global_frameworks()
        if frameworks:
            framework_names = list(frameworks.keys())
        else:
            framework_names = ["No segmenters available"]
        return framework_names

    def __str__(self) -> str:
        return f"ImageDataModel({self.parent_directory}, {self.get_image_count()} images)"
