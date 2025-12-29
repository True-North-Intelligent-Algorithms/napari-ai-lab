"""
Stacked Sequence Writer for label storage.

This writer loads a directory of TIFFs as a stacked array (for viewing),
but saves individual TIFF files (maintaining directory structure).
"""

from pathlib import Path

import numpy as np
from skimage import io

from ..utility import collect_all_image_names, get_axis_info, pad_to_largest
from .base_writer import BaseWriter


class StackedSequenceWriter(BaseWriter):
    """
    Writer that loads directory as stack but saves individual TIFFs.

    This maintains the same file storage (individual TIFF files) while
    providing a stacked view for the napari viewer.
    """

    def __init__(self, subdirectory: str = "class_0"):
        """
        Initialize the stacked sequence writer.

        Args:
            subdirectory: Subdirectory name under 'annotations' (default: "class_0")
        """
        super().__init__(subdirectory)
        self._cached_stack = None
        self._cached_directory = None
        self._image_names = []
        self._original_shapes = []
        self._normalize = False

    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
    ) -> bool:
        """Save data as individual TIFF, cropped to original size."""
        try:
            # Use current_step[0] to get index in stack
            if current_step:
                idx = current_step[0]
                # Get actual dataset name from index
                dataset_name = self._image_names[idx]
            else:
                idx = self._image_names.index(dataset_name)

            original_shape = self._original_shapes[idx]
            cropped_data = data[idx, ...]
            cropped_data = np.squeeze(cropped_data)
            if cropped_data.shape != original_shape:
                cropped_data = cropped_data[
                    tuple(slice(0, s) for s in original_shape)
                ]

            path = Path(save_directory) / f"{dataset_name}.tif"
            io.imsave(str(path), cropped_data.astype(np.uint16))
            return True

        except Exception as e:  # noqa: BLE001
            # Catch all errors (IO, index, type conversion)
            print(f"Error saving {dataset_name}: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        """
        Load entire directory as stack, return slice matching dataset_name.

        Args:
            load_directory: Directory where TIFFs are stored
            dataset_name: Name of the dataset (used to find correct index)

        Returns:
            Array data for the specified dataset (empty if no saved data exists)
        """
        image_names = collect_all_image_names(load_directory)
        self.load_image_collection_as_stack(load_directory, image_names)
        return self._cached_stack

    def load_image_collection_as_stack(
        self, directory: str, image_names: list
    ):
        """
        Load images listed in `image_names` from `directory` as a padded stack.

        This function loads files exactly as named in `image_names` using
        skimage.io.imread. It does not try to add or change extensions.
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            self._cached_stack = np.array([])
            self._image_names = []
            self._original_shapes = []
            self._cached_directory = directory
            return

        if not image_names:
            self._cached_stack = np.array([])
            self._image_names = []
            self._original_shapes = []
            self._cached_directory = directory
            return

        print(f"Loading {len(image_names)} files as stack...")

        images = []
        axis_infos = []
        self._image_names = []
        self._original_shapes = []

        for name in image_names:
            path = dir_path / name
            img = io.imread(str(path))
            axis_info = get_axis_info(img)
            images.append(img)
            axis_infos.append(axis_info)
            self._image_names.append(path.stem)
            self._original_shapes.append(img.shape)

        if self._normalize:
            normalized_images = []
            for img in images:
                min_val = np.min(img)
                max_val = np.max(img)
                if max_val > min_val:
                    normalized = (img - min_val) / (max_val - min_val)
                    normalized_images.append(normalized)
                else:
                    normalized_images.append(img)
            images = normalized_images

        self._cached_stack = pad_to_largest(images, axis_infos)
        self._cached_directory = directory
        print(f"Cached stack shape: {self._cached_stack.shape}")

    def load_full_stack(
        self, load_directory: str, normalize: bool = False
    ) -> np.ndarray:
        """
        Load entire directory as a stacked array.

        This is a convenience method for getting the full stack.

        Args:
            load_directory: Directory where TIFFs are stored
            normalize: Whether to normalize each image to same scale (0-1).
                      Use True for images, False for labels/predictions.

        Returns:
            Full stacked array
        """
        # Check if we need to reload due to normalize flag change
        if (
            self._cached_directory != load_directory
            or self._normalize != normalize
        ):
            self._normalize = normalize
            image_names = collect_all_image_names(load_directory)
            self.load_image_collection_as_stack(load_directory, image_names)

        return (
            self._cached_stack
            if self._cached_stack is not None
            else np.array([])
        )
