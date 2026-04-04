"""
Stacked Sequence artifact I/O for storing artifacts.

This artifact I/O loads a directory of images as a stacked array (for viewing),
but saves individual artifact files (maintaining directory structure).
"""

from pathlib import Path

import numpy as np
from skimage import io

from napari_ai_lab.utilities.image_util import compute_collapsed_shape

from ..utility import collect_all_image_names, get_axis_info, pad_to_largest
from .base_artifact_io import BaseArtifactIO


class StackedSequenceArtifactIO(BaseArtifactIO):
    """
    I/O implementation that loads a directory as a stacked array but saves
    individual image files. This preserves file-per-frame storage while
    providing a stacked view for the napari viewer.
    """

    def __init__(self, subdirectory: str = "class_0"):
        super().__init__(subdirectory)
        self._current_stack = None
        self._current_directory = None
        self._image_names = []
        self._original_shapes = []
        self._normalize = False

    def save(
        self,
        save_directory: str,
        dataset_name: str,
        data: np.ndarray,
        current_step: tuple = None,
        selected_axis: str = None,
    ) -> bool:
        try:
            if current_step:
                idx = current_step[0]
                dataset_name = self._image_names[idx]
            else:
                idx = self._image_names.index(dataset_name)

            """
            original_shape = self._original_shapes[idx]
            cropped_data = data[idx, ...]
            cropped_data = np.squeeze(cropped_data)
            if cropped_data.shape != original_shape:
                cropped_data = cropped_data[
                    tuple(slice(0, s) for s in original_shape)
                ]

            path = Path(save_directory) / f"{dataset_name}.tif"
            io.imsave(str(path), cropped_data.astype(np.uint16))
            """
            path = Path(save_directory) / f"{dataset_name}.tif"
            io.imsave(str(path), data.astype(np.uint16))
            return True
        except Exception as e:  # noqa: BLE001
            print(f"Error saving {dataset_name}: {e}")
            return False

    def load(self, load_directory: str, dataset_name: str) -> np.ndarray:
        if len(self._image_names) == 0:
            image_names = collect_all_image_names(load_directory)
            self.load_image_collection_as_stack(
                load_directory, self._image_names
            )
        else:
            image_names = self._image_names
            self.load_sparse_image_collection_as_stack(
                load_directory, image_names
            )
        return self._current_stack

    def load_image_collection_as_stack(
        self, directory: str, image_names: list
    ):
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            self._current_stack = np.array([])
            self._image_names = []
            self._original_shapes = []
            self._current_directory = directory
            return

        if not image_names:
            self._current_stack = np.array([])
            self._image_names = []
            self._original_shapes = []
            self._current_directory = directory
            return

        images = []
        self._axes_infos = []
        self._image_names = []
        self._original_shapes = []

        for name in image_names:
            path = dir_path / name
            img = io.imread(str(path))
            axis_info = get_axis_info(img)
            images.append(img)
            self._axes_infos.append(axis_info)
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

        self._current_stack = pad_to_largest(images, self._axes_infos)
        self._current_directory = directory
        print(f"Cached stack shape: {self._current_stack.shape}")

    def load_sparse_image_collection_as_stack(
        self, directory: str, image_names: list
    ):
        """Load sparse collection - create empty frames for missing files."""
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"Directory does not exist: {directory}")
            self._current_stack = np.array([])
            self._current_directory = directory
            return

        if not image_names or not self._original_shapes:
            self._current_stack = np.array([])
            self._current_directory = directory
            return

        images = []

        for idx, name in enumerate(image_names):
            # Add .tif if no extension
            filename = f"{name}.tif" if "." not in name else name

            path = dir_path / filename

            if path.exists():
                # Load existing file
                img = io.imread(str(path))
                self._axes_infos[idx] = get_axis_info(img)
            else:
                # if an axis is collapsed (not in output) then create empty frame using collapsed shape
                if self._axes_to_collapse is not None:
                    collapsed_shape = compute_collapsed_shape(
                        self._original_shapes[idx],
                        self._axes_infos[idx],
                        self.axes_to_collapse,
                    )
                    self._axes_infos[idx] = self._axes_infos[idx].replace(
                        self.axes_to_collapse, ""
                    )
                # Otherwise create empty frame using original shape
                else:
                    collapsed_shape = self._original_shapes[idx]

                img = np.zeros(collapsed_shape, dtype=np.uint16)

            images.append(img)

        self._current_stack = pad_to_largest(images, self._axes_infos)
        self._current_directory = directory
        print(f"Cached sparse stack shape: {self._current_stack.shape}")

    def load_full_stack(
        self, load_directory: str, normalize: bool = False
    ) -> np.ndarray:
        if (
            self._current_directory != load_directory
            or self._normalize != normalize
        ):
            self._normalize = normalize
            image_names = collect_all_image_names(load_directory)
            self.load_image_collection_as_stack(load_directory, image_names)

        return (
            self._current_stack
            if self._current_stack is not None
            else np.array([])
        )

    def get_image_names(self):
        """Get list of image names."""
        return self._image_names

    def get_original_shapes(self):
        """Get list of original image shapes."""
        return self._original_shapes

    def set_image_names(self, image_names: list):
        """Set image names."""
        self._image_names = image_names

    def set_original_shapes(self, original_shapes: list):
        """Set original shapes."""
        self._original_shapes = original_shapes

    def get_axes_infos(self):
        """Get list of axis infos."""
        return self._axes_infos

    def set_axes_infos(self, axes_infos: list):
        """Set list of axis infos."""
        self._axes_infos = axes_infos

    def set_axes_to_collapse(self, axes_to_collapse: str | list[str] | None):
        """Set axes to collapse when loading/saving."""
        self._axes_to_collapse = axes_to_collapse

    @property
    def axes_to_collapse(self):
        """Get axes to collapse."""
        return getattr(self, "_axes_to_collapse", None)
