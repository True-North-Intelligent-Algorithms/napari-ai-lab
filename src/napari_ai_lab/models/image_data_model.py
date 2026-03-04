"""
Simple ImageDataModel for managing image directories and file organization.

This model handles:
- Image directory scanning
- Result directory organization
- Path generation for different result types
"""

from pathlib import Path

import numpy as np
from skimage.io import imread

from napari_ai_lab.utility import (
    collect_all_image_names,
    create_empty_instance_image,
    get_axis_info_from_shape,
    remove_trivial_axes,
)

from ..artifact_io import get_artifact_io


class ImageDataModel:
    """
    Simple model for managing image data and result organization.

    Handles image directory scanning and provides organized paths
    for saving different types of results (segmentations, embeddings, etc.).
    """

    def __init__(self, parent_directory: str):
        """
        Initialize with a parent directory containing images.

        Args:
            parent_directory: Path to directory containing images
        """
        self.parent_directory = Path(parent_directory).resolve()
        self.results_directory = self.parent_directory / "results"
        self._image_paths: list[Path] | None = None
        self.segmenter_cache: dict = {}
        self.annotation_io_type: str = "tiff"
        self.prediction_io_type: str = "tiff"
        self._annotations_io = None
        self._predictions_io = None
        self.axis_types: str | None = None

        # Load image list on initialization
        self._populate_image_list(str(self.parent_directory))

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
        # Centralized image name collection
        self._image_paths = collect_all_image_names(self.parent_directory)

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

    def load_image(self, image_index: int) -> np.ndarray:
        """
        Load image data at the specified index.

        Args:
            image_index: Index of the image to load

        Returns:
            numpy array with image data, squeezed to remove singleton dimensions

        Raises:
            IndexError: If image_index is out of range
            OSError: If image cannot be loaded
        """

        image_paths = self.get_image_paths()
        if not 0 <= image_index < len(image_paths):
            raise IndexError(
                f"Image index {image_index} out of range (0-{len(image_paths)-1})"
            )

        image_path = image_paths[image_index]
        print(f"Loading image: {image_path}")

        # TODO: refactor to use io classes and eventually ndev-io
        if image_path.suffix.lower() == ".czi":
            # Use czifile for .czi format
            from czifile import CziFile

            with CziFile(str(image_path)) as czi:
                self.image_data = czi.asarray()
                self.axis_types = czi.axes
        else:
            # Use skimage for other formats
            self.image_data = imread(str(image_path))
            self.axis_types = get_axis_info_from_shape(self.image_data.shape)

        # Squeeze to remove singleton dimensions
        if self.axis_types and len(self.axis_types) == len(
            self.image_data.shape
        ):
            # Remove axis characters corresponding to trivial dimensions
            self.axis_types = remove_trivial_axes(
                self.axis_types, self.image_data.shape
            )

        self.image_data = np.squeeze(self.image_data)
        print(f"Loaded image shape: {self.image_data.shape}")
        print(f"Axis types: {self.axis_types}")

        return self.image_data

    def _populate_image_list(self, directory: str):
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

    def get_annotations_directory(self, subdirectory: str = "class_0") -> Path:
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

    def get_patches_directory(self, axis: int | None = None) -> Path:
        """
        Get the base directory for storing patches.

        Args:
            axis: Optional axis identifier to append to directory name.
                  If None, returns base patches directory.
                  If specified, returns patches/patches_axis_{axis}/ directory.

        Returns:
            Path to patches/ folder or patches/patches_axis_{axis}/ folder
        """
        if axis is None:
            patches_dir = self.parent_directory / "patches"
        else:
            patches_dir = (
                self.parent_directory / "patches" / f"patches_axis_{axis}"
            )

        patches_dir.mkdir(parents=True, exist_ok=True)

        return patches_dir

    def delete_patches(self, axis: int | None = None):
        """
        Delete all patch files in the patches directory.

        Args:
            axis: Optional axis identifier to specify which patches directory to clear.
                  If None, clears the base patches/ directory.
                  If specified, clears patches/patches_axis_{axis}/ directory.
        """
        patches_dir = self.get_patches_directory(axis)

        if patches_dir.exists() and patches_dir.is_dir():
            import shutil

            shutil.rmtree(patches_dir)
        else:
            print(f"Patches directory does not exist: {patches_dir}")

    def get_models_directory(self) -> Path:
        """
        Get the directory for storing trained models.

        Returns:
            Path to models/ folder
        """
        models_dir = self.parent_directory / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    def generate_model_name(self, segmenter) -> str:
        """
        Generate a unique model name based on segmenter type and current datetime.

        Args:
            segmenter: The segmenter instance to generate a name for

        Returns:
            str: Model name in format <SegmenterClassName>_YYYYMMDD_HHMMSS.pth
        """
        from datetime import datetime

        # Generate model name: <SegmenterClassName>_YYYYMMDD_HHMMSS.pth
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        segmenter_class_name = segmenter.__class__.__name__
        model_name = f"{segmenter_class_name}_{timestamp}.pth"
        return model_name

    def get_parent_directory(self) -> Path:
        """
        Get the parent directory containing images.

        Returns:
            Path to parent directory
        """
        return self.parent_directory

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

    def get_annotations_io(self):
        """Get annotation artifact io."""
        if self._annotations_io is None:
            from ..artifact_io import get_artifact_io

            self._annotations_io = get_artifact_io(self.annotation_io_type)
        return self._annotations_io

    def set_annotation_io_type(self, io_type: str):
        """Set annotation artifact io type."""
        self.annotation_io_type = io_type
        self._annotations_io = None

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
            numpy ndarray containing annotation labels (dtype preserved by io or uint16 zeros).
        """
        # Defer imports to here to avoid bringing numpy/io into top-level module load
        import numpy as np

        annotation_dir = self.get_annotations_directory(subdirectory)

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            dataset_name = image_paths[image_index].stem
        else:
            dataset_name = "unknown"

        io = self.get_annotations_io()
        data = io.load(str(annotation_dir), dataset_name)

        # If nothing saved, return zeros
        if data is None or getattr(data, "size", 0) == 0:
            # Create an empty instance image using centralized helper so
            # annotations and predictions share the same shape rules.
            return create_empty_instance_image(image_shape, dtype=np.uint16)

        return data

    def get_predictions_io(self):
        """Get prediction artifact io."""
        return self._predictions_io

    def set_prediction_io_type(self, io_type: str, axis_slice: str = None):
        """Set prediction artifact io type."""
        self.prediction_io_type = io_type
        self._predictions_io = get_artifact_io(self.prediction_io_type)
        if axis_slice is not None and hasattr(
            self._predictions_io, "set_axis_slice"
        ):
            self._predictions_io.set_axis_slice(axis_slice)

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
            numpy ndarray containing prediction labels (dtype preserved by io or uint16 zeros).
        """
        import numpy as np

        preds_dir = self.get_predictions_directory(
            subdirectory if subdirectory else "predictions"
        )

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            dataset_name = image_paths[image_index].stem
        else:
            dataset_name = "unknown"

        io = self.get_predictions_io()

        if hasattr(io, "set_shape_total"):
            io.set_shape_total(self.image_data.shape)

        data = io.load(str(preds_dir), dataset_name)

        if data is None or getattr(data, "size", 0) == 0:
            return create_empty_instance_image(image_shape, dtype=np.uint16)

        return data

    def save_annotations(
        self,
        labels_array,
        image_index: int,
        subdirectory: str = "class_0",
        current_step: tuple = None,
    ):
        """
        Save the provided labels array for the image at image_index under the
        annotations/subdirectory directory.

        Args:
            labels_array: numpy array containing labels to save.
            image_index: Index of the image to associate the labels with.
            subdirectory: Subdirectory under annotations to save into (default: class_0).
            current_step: Viewer dimension position (for stacked mode).

        Returns:
            The result of the io.save(...) call.
        """
        annotation_dir = self.get_annotations_directory(subdirectory)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.get_image_paths()

        dataset_name = image_paths[image_index].stem

        io = self.get_annotations_io()

        # Ensure uint16 to match previous behavior
        labels_to_save = np.asarray(labels_array).astype(np.uint16)

        return io.save(
            str(annotation_dir), dataset_name, labels_to_save, current_step
        )

    def save_predictions(
        self,
        predictions_array,
        image_index: int,
        subdirectory: str = "predictions",
        current_step: tuple = None,
        selected_axis: str = None,
    ):
        """
        Save prediction array for the image at image_index under predictions/subdirectory.

        Args:
            preds_array: numpy array with predictions to save.
            image_index: Index of the associated image.
            subdirectory: Subdirectory under predictions to save into.
            current_step: Viewer dimension position (for stacked mode).
            selected_axis: Axis string like "YX", "ZYX", "YXC", etc.

        Returns:
            Result of io.save(...)
        """
        import numpy as np

        predictions_dir = self.get_predictions_directory(subdirectory)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            dataset_name = image_paths[image_index].stem
        else:
            dataset_name = "unknown"

        io = self.get_predictions_io()

        predictions_to_save = np.asarray(predictions_array)

        return io.save(
            str(predictions_dir),
            dataset_name,
            predictions_to_save,
            current_step,
            selected_axis,
        )

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

    def get_segmenter(self, segmenter_name: str):
        """
        Return a cached segmenter instance by name, creating and caching it if needed.
        """
        # Return from cache if available
        if segmenter_name in self.segmenter_cache:
            return self.segmenter_cache[segmenter_name]

        # Try Interactive segmenters first
        try:
            from ..Segmenters.InteractiveSegmenters import (
                InteractiveSegmenterBase,
            )

            seg_cls = InteractiveSegmenterBase.get_framework(segmenter_name)
            if seg_cls:
                inst = seg_cls()
                self.segmenter_cache[segmenter_name] = inst
                return inst
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # Import or construction failed; return None and allow caller to handle
            pass

        # Try Global segmenters
        try:
            from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase

            seg_cls = GlobalSegmenterBase.get_framework(segmenter_name)
            if seg_cls:
                inst = seg_cls()
                self.segmenter_cache[segmenter_name] = inst
                return inst
        except (ImportError, AttributeError, TypeError, RuntimeError):
            # Import or construction failed; return None and allow caller to handle
            pass

        return None

    def set_augmenter(self, augmenter):
        """
        Set the augmenter to use for patch generation.

        Args:
            augmenter: An augmenter instance (SimpleAugmenter, AlbumentationsAugmenter, etc.)
        """
        self.augmenter = augmenter

    def set_patch_size(self, patch_size: tuple[int, ...]):
        """
        Set the patch size for augmentation.

        Args:
            patch_size: Tuple specifying patch dimensions (e.g., (128, 128) or (1, 128, 128))
        """
        self.patch_size = patch_size

    def set_num_patches(self, num_patches: int):
        """
        Set the number of patches to generate.

        Args:
            num_patches: Number of patches to generate during augmentation
        """
        self.num_patches = num_patches

    def setup_augmentation(
        self,
        image: np.ndarray,
        annotations: np.ndarray,
        mode: str = "valid_coordinates",
        compute_global_stats: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ):
        """
        Setup augmentation by computing normalization stats and/or valid coordinates.

        Args:
            image: Full image array
            annotations: Full annotations array
            mode: Augmentation mode - "valid_coordinates", "marked_roi", or "random_crop"
                  (currently only "valid_coordinates" is implemented)
            compute_global_stats: Whether to compute global normalization statistics
            percentile_low: Lower percentile for normalization (default: 1.0)
            percentile_high: Upper percentile for normalization (default: 99.0)

        Raises:
            ValueError: If augmenter or patch_size not set
            NotImplementedError: If mode is not "valid_coordinates"
        """
        if not hasattr(self, "augmenter") or self.augmenter is None:
            raise ValueError("Augmenter not set. Call set_augmenter() first.")

        if not hasattr(self, "patch_size") or self.patch_size is None:
            raise ValueError(
                "Patch size not set. Call set_patch_size() first."
            )

        # Compute global normalization statistics if requested
        if compute_global_stats and hasattr(
            self.augmenter, "compute_global_normalization_stats"
        ):
            print(
                "Computing global normalization statistics from full image..."
            )
            self.augmenter.compute_global_normalization_stats(
                image, percentile_low, percentile_high
            )

        # Setup augmentation mode
        if mode == "valid_coordinates":
            if hasattr(self.augmenter, "create_valid_coordinates"):
                print("Computing valid coordinates for patches...")
                self.augmenter.create_valid_coordinates(
                    annotations, image.shape, self.patch_size, axis=None
                )
                if hasattr(self.augmenter, "valid_coordinates"):
                    print(
                        f"Found {len(self.augmenter.valid_coordinates)} valid positions"
                    )
            else:
                print(
                    "Warning: Augmenter does not support create_valid_coordinates"
                )
        elif mode == "marked_roi":
            raise NotImplementedError(
                "marked_roi mode not yet implemented. Use 'valid_coordinates' or 'random_crop'."
            )
        elif mode == "random_crop":
            # Random crop mode doesn't need special setup
            print("Using random crop mode (no coordinate pre-computation)")
        else:
            raise ValueError(
                f"Unknown augmentation mode: {mode}. Use 'valid_coordinates', 'marked_roi', or 'random_crop'."
            )

    def generate_patches(
        self,
        image: np.ndarray,
        annotations: np.ndarray,
        axis: str | None = None,
        patch_base_name: str = "patch",
        axes_string: str = "YX",
        num_inputs: int = 1,
        num_truths: int = 1,
        sub_sample: int = 1,
    ) -> Path:
        """
        Generate augmented patches and save them to the patches directory.

        Args:
            image: Full image array
            annotations: Full annotations array
            axis: Optional axis identifier for patches directory (e.g., "yx")
            patch_base_name: Base name for patch files (default: "patch")
            axes_string: String describing axes for info.json (e.g., "YX", "ZYX")
            num_inputs: Number of input channels for info.json (default: 1)
            num_truths: Number of truth classes for info.json (default: 1)
            sub_sample: Subsampling factor for info.json (default: 1)

        Returns:
            Path to the patches directory

        Raises:
            ValueError: If augmenter, patch_size, or num_patches not set
        """
        if not hasattr(self, "augmenter") or self.augmenter is None:
            raise ValueError("Augmenter not set. Call set_augmenter() first.")

        if not hasattr(self, "patch_size") or self.patch_size is None:
            raise ValueError(
                "Patch size not set. Call set_patch_size() first."
            )

        if not hasattr(self, "num_patches") or self.num_patches is None:
            raise ValueError(
                "Number of patches not set. Call set_num_patches() first."
            )

        # Get patches directory
        patches_dir = self.get_patches_directory(axis=axis)

        # Generate patches
        print(f"\nGenerating {self.num_patches} patches...")
        for i in range(self.num_patches):
            self.augmenter.augment_and_save(
                image,
                annotations,
                str(patches_dir),
                patch_base_name,
                self.patch_size,
                axis=None,
            )
            if (i + 1) % 10 == 0 or (i + 1) == self.num_patches:
                print(f"  Created {i+1}/{self.num_patches} patches")

        # Write info.json
        print("\nWriting info.json...")
        if hasattr(self.augmenter, "write_info"):
            self.augmenter.write_info(
                patch_path=str(patches_dir),
                axes=axes_string,
                num_inputs=num_inputs,
                num_truths=num_truths,
                sub_sample=sub_sample,
            )

        print(f"✅ Created {self.num_patches} patches in {patches_dir}")
        return patches_dir

    def segment(self, segmenter, image_slice, points=None, shapes=None):
        """
        Perform segmentation using the provided segmenter.

        This method wraps the segmenter's segment call and automatically
        provides the parent_directory from the model.

        Args:
            segmenter: The segmenter instance to use
            image_slice: The image data to segment
            points: Optional points for interactive segmentation
            shapes: Optional shapes for interactive segmentation

        Returns:
            numpy.ndarray: Segmentation mask
        """
        return segmenter.segment(
            image_slice,
            points=points,
            shapes=shapes,
            parent_directory=self.parent_directory,
        )

    def __str__(self) -> str:
        return f"ImageDataModel({self.parent_directory}, {self.get_image_count()} images)"
