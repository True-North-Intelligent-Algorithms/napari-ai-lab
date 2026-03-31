"""
Simple ImageDataModel for managing image directories and file organization.

This model handles:
- Image directory scanning
- Result directory organization
- Path generation for different result types
- Annotation and prediction shape mapping (collapsing axes as needed)

Annotation/Prediction Axis Collapsing
--------------------------------------
When working with multi-dimensional images, annotations and predictions may need
different dimensionality than the source image. For example:
- Image: ZYXC (3D + channels) -> Annotations/Predictions: ZYX (3D only)
- Image: TYXC (time + channels) -> Annotations/Predictions: TYX (time, no channels)

Usage example:
    # For ZYXC image, get ZYX annotations (collapse channels)
    labels = model.load_existing_annotations(
        image_shape=(10, 512, 512, 3),
        image_index=0,
        axes_to_collapse="C"
    )
    # Returns shape (10, 512, 512)

    # Same for predictions
    predictions = model.load_existing_predictions(
        image_shape=(10, 512, 512, 3),
        image_index=0,
        axes_to_collapse="C"
    )
    # Returns shape (10, 512, 512)

    # For future: collapse multiple axes
    labels = model.load_existing_annotations(
        image_shape=(5, 10, 512, 512, 3),
        image_index=0,
        axes_to_collapse=["T", "C"]  # Collapse time and channels
    )
    # Returns shape (10, 512, 512)
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
        self.input_images_io_type: str | None = None
        self._annotations_io = None
        self._predictions_io = None
        self._input_images_io = None
        self._current_segmenter_name: str | None = None
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
            algorithm: Optional algorithm/method name to create subdirectory for organizing
                      predictions by segmentation method (e.g., "StarDist", "Cellpose").
                      If None, returns base predictions/ folder.

        Returns:
            Path to predictions/ folder or predictions/{algorithm}/ subfolder
        """
        if algorithm:
            return self.parent_directory / "predictions" / algorithm
        return self.parent_directory / "predictions"

    def _detect_artifact_io_type(
        self, artifact_dir: Path, default_type: str = "tiff"
    ) -> str:
        """
        Detect artifact IO type by examining existing files in directory.

        Args:
            artifact_dir: Directory to scan for artifacts
            default_type: Default type if no files found (default: "tiff")

        Returns:
            Detected IO type string ("tiff", "zarr", "numpy", "tiff_slice", "stacked_sequence")
        """
        if not artifact_dir.exists():
            return default_type

        # Check for different file types in order of specificity
        # Zarr stores are directories with .zarr extension or .zarray file
        if list(artifact_dir.glob("**/*.zarr")) or list(
            artifact_dir.glob("**/.zarray")
        ):
            return "zarr"

        # Numpy files (.npy)
        if list(artifact_dir.glob("**/*.npy")):
            return "numpy"

        # TIFF files (.tif, .tiff)
        if list(artifact_dir.glob("**/*.tif")) or list(
            artifact_dir.glob("**/*.tiff")
        ):
            # Could be tiff or tiff_slice - for now return "tiff"
            # TODO: Add more sophisticated detection if needed
            return "tiff"

        # No files found, use default
        return default_type

    def get_annotations_io(self):
        """Get annotation artifact io, auto-detecting type if not set."""
        if self._annotations_io is None:
            # Auto-detect IO type from existing files if not explicitly set
            annotation_dir = self.parent_directory / "annotations"
            detected_type = self._detect_artifact_io_type(
                annotation_dir, self.annotation_io_type
            )

            # Update type if detected different from current
            if detected_type != self.annotation_io_type:
                print(f"📁 Auto-detected annotation IO type: {detected_type}")
                self.annotation_io_type = detected_type

            self._annotations_io = get_artifact_io(self.annotation_io_type)

        # Transfer metadata from input_images_io if both are stacked_sequence
        if (
            self.annotation_io_type == "stacked_sequence"
            and self.input_images_io_type == "stacked_sequence"
            and self._input_images_io is not None
        ):
            image_names = self._input_images_io.get_image_names()
            original_shapes = self._input_images_io.get_original_shapes()
            self._annotations_io.set_image_names(image_names)
            self._annotations_io.set_original_shapes(original_shapes)

        return self._annotations_io

    def set_annotation_io_type(self, io_type: str):
        """Set annotation artifact io type."""
        self.annotation_io_type = io_type
        self._annotations_io = None

    def _compute_annotation_shape(
        self,
        image_shape: tuple,
        axes_to_collapse: str | list[str] | None = None,
    ) -> tuple:
        """
        Compute annotation shape by collapsing specified axes from image shape.

        Simple, flexible approach: caller specifies which axes to remove.
        Today: collapse "C" for ZYXC -> ZYX
        Future: collapse "T", ["C", "S"], or any other axes as needed

        Args:
            image_shape: Original image shape
            axes_to_collapse: Axis names to remove (e.g., "C" or ["C", "T"])
                            If None, annotation shape matches image shape

        Returns:
            Tuple representing the annotation shape with specified axes collapsed
        """
        if axes_to_collapse is None or not self.axis_types:
            return image_shape

        # Normalize to list
        if isinstance(axes_to_collapse, str):
            axes_to_collapse = [axes_to_collapse]

        # Build new shape by keeping only non-collapsed axes
        new_shape = []
        for axis_name, dim_size in zip(
            self.axis_types, image_shape, strict=False
        ):
            if axis_name not in axes_to_collapse:
                new_shape.append(dim_size)

        return tuple(new_shape)

    def load_existing_annotations(
        self,
        image_shape,
        image_index: int = 0,
        subdirectory: str = "class_0",
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Load existing annotation array for the image at image_index, or return an
        empty array matching annotation shape if no saved data exists.

        Args:
            image_shape: Shape of the image to match for empty array creation.
            image_index: Index of the image in the model's image list.
            subdirectory: Subdirectory under 'annotations' to look in (default: class_0).
            axes_to_collapse: Axis names to collapse from image shape (e.g., "C" for channels).
                            Pass "C" to get ZYX annotations from ZYXC image.
                            Pass ["C", "T"] to collapse multiple axes.
                            Pass None to match image shape exactly.

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

        # Compute target annotation shape (may be smaller than image if axes collapsed)
        annotation_shape = self._compute_annotation_shape(
            image_shape, axes_to_collapse
        )

        # If nothing saved, return zeros with appropriate shape
        if data is None or getattr(data, "size", 0) == 0:
            # Create an empty instance image using centralized helper so
            # annotations and predictions share the same shape rules.
            return create_empty_instance_image(
                annotation_shape, dtype=np.uint16
            )

        return data

    def get_predictions_io(self):
        """Get prediction artifact io, auto-detecting type if not set."""
        # Use segmenter name as subdirectory
        subdirectory = self._current_segmenter_name or "default"

        # Create IO with subdirectory
        self._predictions_io = get_artifact_io(
            self.prediction_io_type, subdirectory=subdirectory
        )

        # Transfer metadata from input_images_io if both are stacked_sequence
        if (
            self.prediction_io_type == "stacked_sequence"
            and self.input_images_io_type == "stacked_sequence"
            and self._input_images_io is not None
        ):
            image_names = self._input_images_io.get_image_names()
            original_shapes = self._input_images_io.get_original_shapes()
            self._predictions_io.set_image_names(image_names)
            self._predictions_io.set_original_shapes(original_shapes)

        return self._predictions_io

    def set_prediction_io_type(self, io_type: str, axis_slice: str = None):
        """Set prediction artifact io type."""
        self.prediction_io_type = io_type
        self._predictions_io = get_artifact_io(self.prediction_io_type)
        if axis_slice is not None and hasattr(
            self._predictions_io, "set_axis_slice"
        ):
            self._predictions_io.set_axis_slice(axis_slice)

    def set_current_segmenter_name(self, name: str):
        """Set current segmenter name for organizing predictions."""
        if name != self._current_segmenter_name:
            self._current_segmenter_name = name
            # Reset predictions_io to use new subdirectory
            self._predictions_io = None

    def get_input_images_io(self):
        """Get input images io."""
        if self._input_images_io is None and self.input_images_io_type:
            self._input_images_io = get_artifact_io(self.input_images_io_type)
        return self._input_images_io

    def set_input_images_io_type(self, io_type: str):
        """Set input images io type."""
        self.input_images_io_type = io_type
        self._input_images_io = get_artifact_io(self.input_images_io_type)

    def load_existing_predictions(
        self,
        image_shape,
        image_index: int = 0,
        subdirectory: str | None = None,
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Load existing prediction array for the image at image_index, or return an
        empty array matching prediction shape if no saved data exists.

        Args:
            image_shape: Shape of the image to match for empty array creation.
            image_index: Index of the image in the model's image list.
            subdirectory: Subdirectory under 'predictions' to look in.
                         If None, uses _current_segmenter_name or "default".
            axes_to_collapse: Axis names to collapse from image shape (e.g., "C" for channels).
                            Pass "C" to get ZYX predictions from ZYXC image.
                            Pass ["C", "T"] to collapse multiple axes.
                            Pass None to match image shape exactly.

        Returns:
            numpy ndarray containing prediction labels (dtype preserved by io or uint16 zeros).
        """
        import numpy as np

        # Use provided subdirectory, or fall back to current segmenter name, or "default"
        if subdirectory is None:
            subdirectory = self._current_segmenter_name or "default"

        preds_dir = self.get_predictions_directory(subdirectory)

        image_paths = self.get_image_paths()
        if 0 <= image_index < len(image_paths):
            dataset_name = image_paths[image_index].stem
        else:
            dataset_name = "unknown"

        io = self.get_predictions_io()

        if hasattr(io, "set_shape_total"):
            io.set_shape_total(self.image_data.shape)

        data = io.load(str(preds_dir), dataset_name)

        # Compute target prediction shape (may be smaller than image if axes collapsed)
        prediction_shape = self._compute_annotation_shape(
            image_shape, axes_to_collapse
        )

        if data is None or getattr(data, "size", 0) == 0:
            return create_empty_instance_image(
                prediction_shape, dtype=np.uint16
            )

        return data

    # ------------------------------------------------------------------
    # Boxes / bounding-box ROI  (saved as CSV in labels/ directory)
    # ------------------------------------------------------------------

    def get_labels_directory(self) -> Path:
        """
        Get the directory for storing label-related CSV files (e.g., boxes).

        Returns:
            Path to labels/ folder (created on demand)
        """
        labels_dir = self.parent_directory / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        return labels_dir

    def get_boxes_csv_path(self) -> Path:
        """Return the path to boxes.csv inside the labels directory."""
        return self.get_labels_directory() / "boxes.csv"

    def _is_stacked_sequence(self) -> bool:
        """Return True when the input images are loaded as a stacked sequence."""
        return self.input_images_io_type == "stacked_sequence"

    def _get_stacked_image_names(self) -> list[str]:
        """
        Return the ordered list of image *stems* from the stacked-sequence IO.

        Returns an empty list when not in stacked_sequence mode or the IO has
        not been initialised yet.
        """
        if (
            self._input_images_io is not None
            and self.input_images_io_type == "stacked_sequence"
        ):
            names = self._input_images_io.get_image_names()
            return list(names) if names else []
        return []

    def save_boxes(
        self,
        boxes_layer_data: list,
        image_index: int,
    ) -> bool:
        """
        Save all boxes in *boxes_layer_data* to ``labels/boxes.csv``.

        **Normal mode** (one image per viewer slot)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Every box in *boxes_layer_data* belongs to the image identified by
        *image_index*.  ``file_name`` is set to
        ``image_paths[image_index].name``.

        **Stacked-sequence mode** (many images stacked on axis 0)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Each box vertex array has shape ``(N, 3)`` where column 0 is the
        **frame/sequence index** within the stack.  The model uses that index
        to look up the original file name from the stacked-sequence IO so that
        each row in the CSV correctly records which source image the box
        belongs to.  All previously saved rows for *all* images that appear
        in the current stack are replaced; rows for images not in the current
        stack are preserved.

        CSV columns::

            file_name, xstart, ystart, xend, yend

        Args:
            boxes_layer_data: ``boxes_layer.data`` — list of vertex arrays.
            image_index: Index of the current image in non-stacked mode
                         (ignored for per-box lookup in stacked mode, but
                         used to replace the right rows when the list is empty).

        Returns:
            True on success.
        """
        import csv

        import numpy as np

        csv_path = self.get_boxes_csv_path()
        fieldnames = ["file_name", "xstart", "ystart", "xend", "yend"]
        stacked = self._is_stacked_sequence()
        stacked_names = self._get_stacked_image_names() if stacked else []

        # ------------------------------------------------------------------ #
        # Determine which file_name(s) will be replaced by this save so we   #
        # know which existing rows to discard.                                #
        # ------------------------------------------------------------------ #
        if stacked:
            # Replace rows for every image in the current stack
            names_to_replace: set[str] = set(stacked_names)
        else:
            image_paths = self.get_image_paths()
            if not (0 <= image_index < len(image_paths)):
                print(f"⚠️  save_boxes: image_index {image_index} out of range")
                return False
            names_to_replace = {image_paths[image_index].name}

        # Keep rows that belong to images *outside* the current replace set
        existing_rows: list[dict] = []
        if csv_path.exists():
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("file_name") not in names_to_replace:
                        existing_rows.append(row)

        # ------------------------------------------------------------------ #
        # Build new rows from every box in boxes_layer_data                  #
        # ------------------------------------------------------------------ #
        new_rows: list[dict] = []
        for box in boxes_layer_data:
            box = np.asarray(box)

            if stacked and box.shape[-1] >= 3:
                # Column 0 is the frame/sequence index
                frame_idx = int(round(float(box[0, 0])))
                if 0 <= frame_idx < len(stacked_names):
                    file_name = f"{stacked_names[frame_idx]}.tif"
                else:
                    print(
                        f"⚠️  save_boxes: frame index {frame_idx} out of range "
                        f"(stack has {len(stacked_names)} frames) — skipping box"
                    )
                    continue
            else:
                # Normal mode — all boxes belong to the current image
                image_paths = self.get_image_paths()
                file_name = image_paths[image_index].name

            # Coordinates: last two columns are always (y, x)
            ystart = int(np.min(box[:, -2]))
            yend = int(np.max(box[:, -2]))
            xstart = int(np.min(box[:, -1]))
            xend = int(np.max(box[:, -1]))
            new_rows.append(
                {
                    "file_name": file_name,
                    "xstart": xstart,
                    "ystart": ystart,
                    "xend": xend,
                    "yend": yend,
                }
            )

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows + new_rows)

        print(f"📦 Saved {len(new_rows)} box(es) → {csv_path}")
        return True

    def load_existing_boxes(self) -> list[dict]:
        """
        Load all bounding-box rows from boxes.csv.

        Each returned dict always contains::

            file_name, xstart, ystart, xend, yend

        In **stacked-sequence mode** an additional key ``frame_index`` is
        included — the position of the image within the current stack — so
        callers can place the rectangle at the correct frame when rebuilding
        the boxes layer.  Images whose file name is not found in the current
        stack receive ``frame_index = None`` and should be skipped.

        Returns:
            List of dicts (empty if the file does not exist).
        """
        import csv

        csv_path = self.get_boxes_csv_path()
        if not csv_path.exists():
            return []

        stacked = self._is_stacked_sequence()
        stacked_names = self._get_stacked_image_names() if stacked else []

        # Build a stem → index lookup for fast resolution
        stem_to_idx: dict[str, int] = {}
        if stacked:
            for idx, name in enumerate(stacked_names):
                # names are stems; also accept full filename with extension
                stem_to_idx[name] = idx
                stem_to_idx[Path(name).stem] = idx

        rows: list[dict] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entry: dict = {
                    "file_name": row["file_name"],
                    "xstart": int(row["xstart"]),
                    "ystart": int(row["ystart"]),
                    "xend": int(row["xend"]),
                    "yend": int(row["yend"]),
                }
                if stacked:
                    # Resolve frame index from file name stem
                    stem = Path(row["file_name"]).stem
                    entry["frame_index"] = stem_to_idx.get(
                        stem, stem_to_idx.get(row["file_name"])
                    )
                rows.append(entry)
        return rows

    def get_boxes_for_image(self, image_name: str) -> dict | None:
        """
        Return the saved bounding-box row for a specific image, or None.

        Args:
            image_name: File name to look up (e.g. ``"image1.jpg"``).

        Returns:
            Dict with keys ``xstart, ystart, xend, yend``, or None if not found.
        """
        for row in self.load_existing_boxes():
            if row["file_name"] == image_name:
                return {
                    "xstart": row["xstart"],
                    "ystart": row["ystart"],
                    "xend": row["xend"],
                    "yend": row["yend"],
                }
        return None

    def save_annotations(
        self,
        labels_array,
        image_index: int,
        subdirectory: str = "class_0",
        current_step: tuple = None,
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Save the provided labels array for the image at image_index under the
        annotations/subdirectory directory.

        Args:
            labels_array: numpy array containing labels to save.
            image_index: Index of the image to associate the labels with.
            subdirectory: Subdirectory under annotations to save into (default: class_0).
            current_step: Viewer dimension position (for stacked mode).
            axes_to_collapse: Axis names that were collapsed (for documentation).
                            Should match what was passed to load_existing_annotations.
                            Not used during save, but kept for API consistency.

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
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Save prediction array for the image at image_index under predictions/subdirectory.

        Args:
            preds_array: numpy array with predictions to save.
            image_index: Index of the associated image.
            subdirectory: Subdirectory under predictions to save into.
            current_step: Viewer dimension position (for stacked mode).
            selected_axis: Axis string like "YX", "ZYX", "YXC", etc.
            axes_to_collapse: Axis names that were collapsed (for documentation).
                            Should match what was passed to load_existing_predictions.
                            Not used during save, but kept for API consistency.

        Returns:
            Result of io.save(...)
        """
        import numpy as np

        predictions_dir = self.get_predictions_directory(subdirectory)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.get_image_paths()

        if self.prediction_io_type == "stacked_sequence":
            dataset_name = image_paths[current_step[0]].stem
        else:
            dataset_name = image_paths[image_index].stem

        io = self.get_predictions_io()

        predictions_to_save = np.asarray(predictions_array)

        return io.save(
            str(predictions_dir),
            dataset_name,
            predictions_to_save,
            current_step,
            selected_axis,
        )

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
        progress_logger=None,
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
            progress_logger: Optional ProgressLogger for tracking progress and logging.
                           If None, falls back to print statements.

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

        # Generate patches with progress tracking
        if progress_logger:
            progress_logger.log_info(
                f"🎨 Generating {self.num_patches} patches..."
            )
        else:
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

            # Update progress
            if progress_logger:
                progress_logger.update_progress(
                    i + 1, self.num_patches, "Generating patches"
                )
            elif (i + 1) % 10 == 0 or (i + 1) == self.num_patches:
                print(f"  Created {i+1}/{self.num_patches} patches")

        # Write info.json
        if progress_logger:
            progress_logger.log_info(
                "📝 Writing patch metadata (info.json)..."
            )
        else:
            print("\nWriting info.json...")

        if hasattr(self.augmenter, "write_info"):
            self.augmenter.write_info(
                patch_path=str(patches_dir),
                axes=axes_string,
                num_inputs=num_inputs,
                num_truths=num_truths,
                sub_sample=sub_sample,
            )

        # Log completion
        if progress_logger:
            progress_logger.log_info(
                f"✅ Created {self.num_patches} patches in {patches_dir}"
            )
        else:
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
