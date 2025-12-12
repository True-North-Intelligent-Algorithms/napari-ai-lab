"""
Stacked Sequence Writer for label storage.

This writer loads a directory of TIFFs as a stacked array (for viewing),
but saves individual TIFF files (maintaining directory structure).
"""

from pathlib import Path

import numpy as np
from skimage import io

from ..utility import get_axis_info, pad_to_largest
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
        self._image_names = None
        self._normalize = False  # Whether to normalize images

    def save(
        self, save_directory: str, image_name: str, data: np.ndarray
    ) -> bool:
        """
        Save data as individual TIFF file.

        NOTE: Save functionality disabled for now - not ready yet.

        Args:
            save_directory: Directory where data should be saved
            image_name: Name of the image (without extension)
            data: Array to save (single slice)

        Returns:
            True if successful, False otherwise
        """
        print(
            "StackedSequenceWriter.save() - Save functionality disabled for now"
        )
        return False

        # TODO: Implement proper save for stacked sequence
        # try:
        #     path = Path(save_directory) / f"{image_name}.tif"
        #     io.imsave(str(path), data.astype(np.uint16))
        #
        #     # Invalidate cache since we saved new data
        #     self._cached_stack = None
        #     self._cached_directory = None
        #
        #     return True
        # except (OSError, ValueError, PermissionError) as e:
        #     print(f"✗ Error saving: {e}")
        #     return False

    def load(self, load_directory: str, image_name: str) -> np.ndarray:
        """
        Load entire directory as stack, return slice matching image_name.

        Args:
            load_directory: Directory where TIFFs are stored
            image_name: Name of the image (used to find correct index)

        Returns:
            Array data for the specified image (empty if no saved data exists)
        """
        try:
            self._load_directory_as_stack(load_directory)
            return self._cached_stack
        except (OSError, ValueError, FileNotFoundError) as e:
            print(f"✗ Error loading stack: {e}")
            return np.array([])

    def _load_directory_as_stack(self, directory: str):
        """
        Load all TIFF files in directory as a padded stack.

        Args:
            directory: Directory containing TIFF files
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            self._cached_stack = np.array([])
            self._image_names = []
            self._cached_directory = directory
            return

        # Find all TIFF files
        tiff_files = sorted(
            list(dir_path.glob("*.tif")) + list(dir_path.glob("*.tiff"))
        )

        if not tiff_files:
            print(f"No TIFF files found in {directory}")
            self._cached_stack = np.array([])
            self._image_names = []
            self._cached_directory = directory
            return

        print(f"Loading {len(tiff_files)} annotation files as stack...")

        # Load all images
        images = []
        axis_infos = []
        self._image_names = []

        for tiff_path in tiff_files:
            try:
                img = io.imread(str(tiff_path))
                axis_info = get_axis_info(img)
                images.append(img)
                axis_infos.append(axis_info)
                self._image_names.append(tiff_path.stem)
            except (OSError, ValueError) as e:
                print(f"Error loading {tiff_path}: {e}")
                continue

        if not images:
            self._cached_stack = np.array([])
            self._image_names = []
            self._cached_directory = directory
            return

        # Normalize each image individually if requested
        if self._normalize:
            print("Normalizing images to same scale...")
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

        # Pad and stack images
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
            self._load_directory_as_stack(load_directory)

        return (
            self._cached_stack
            if self._cached_stack is not None
            else np.array([])
        )
