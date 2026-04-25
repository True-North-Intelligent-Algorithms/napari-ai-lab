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
            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)

            if current_step:
                idx = current_step[0]
                file_name = self._image_names[idx]
                path = save_dir / f"{file_name}.tif"
                io.imsave(str(path), data.astype(np.uint16))
                return True

            # Sparse save: iterate the stacked first dimension and write only
            # slices that contain data, cropping out the original (un-padded)
            # shape for each slice.
            n_slices = data.shape[0]
            for n in range(n_slices):
                slice_data = data[n]

                # Crop padding off using the per-slice original shape
                if n < len(self._original_shapes):
                    original_shape = self._original_shapes[n]
                    crop = tuple(slice(0, s) for s in original_shape)
                    slice_data = slice_data[crop]

                # Skip empty (all-zero) slices to keep the save sparse
                if not np.any(slice_data):
                    continue

                file_name = self._image_names[n]
                path = save_dir / f"{file_name}.tif"
                io.imsave(str(path), slice_data.astype(np.uint16))

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
                # Shapes/axes are pre-collapsed by the caller via
                # set_original_shapes_and_axes_infos, so use them directly.
                img = np.zeros(self._original_shapes[idx], dtype=np.uint16)

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

    def set_original_shapes_and_axes_infos(
        self,
        original_shapes: list,
        axes_infos: list,
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Set original shapes and axes infos, collapsing the specified axes.

        When ``axes_to_collapse`` is provided, each shape/axes-info pair is
        reduced by removing the named axes (e.g., collapse ``"C"`` so a
        ``ZYXC`` annotation is stored as ``ZYX``). When ``axes_to_collapse``
        is ``None``, shapes and axes infos are stored as-is.
        """
        collapsed_shapes = []
        collapsed_axes_infos = []
        for shape, axes_info in zip(original_shapes, axes_infos, strict=False):
            if axes_to_collapse is not None and axes_info:
                collapsed_shapes.append(
                    compute_collapsed_shape(shape, axes_info, axes_to_collapse)
                )
                collapse_list = (
                    [axes_to_collapse]
                    if isinstance(axes_to_collapse, str)
                    else list(axes_to_collapse)
                )
                new_axes = "".join(
                    a for a in axes_info if a not in collapse_list
                )
                collapsed_axes_infos.append(new_axes)
            else:
                collapsed_shapes.append(shape)
                collapsed_axes_infos.append(axes_info)

        self._original_shapes = collapsed_shapes
        self._axes_infos = collapsed_axes_infos

    def get_axes_infos(self):
        """Get list of axis infos."""
        return self._axes_infos
