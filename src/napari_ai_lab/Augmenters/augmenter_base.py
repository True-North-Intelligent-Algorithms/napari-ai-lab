from abc import ABC, abstractmethod

import numpy as np


class AugmenterBase(ABC):
    """
    Abstract base class for image and mask augmentation.

    This class provides the interface for augmenting images and their corresponding
    masks, with support for saving augmented patches to disk.

    Attributes
    ----------
    input_dir : str
        Subdirectory name for saving input images (default: "input0")
    ground_truth_dir : str
        Subdirectory name for saving ground truth masks (default: "ground_truth0")
    """

    def __init__(self):
        """Initialize the augmenter with default directory names."""
        self.input_dir = "input0"
        self.ground_truth_dir = "ground_truth0"
        self.valid_coordinates = None

    @abstractmethod
    def augment(
        self,
        im: np.ndarray,
        mask: np.ndarray,
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment an image and its corresponding mask.

        Parameters
        ----------
        im : np.ndarray
            Input image array
        mask : np.ndarray
            Input mask array corresponding to the image
        patch_size : tuple[int, ...]
            Size of the patch to extract/augment
        axis : Optional[int]
            Axis along which to perform augmentation. If None, augment across all axes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Augmented image and mask as a tuple (augmented_image, augmented_mask)
        """

    def augment_and_save(
        self,
        im: np.ndarray,
        mask: np.ndarray,
        patch_path: str,
        patch_base_name: str,
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> tuple[str, str]:
        """
        Augment an image and mask, then save the results to disk.

        Parameters
        ----------
        im : np.ndarray
            Input image array
        mask : np.ndarray
            Input mask array corresponding to the image
        patch_path : str
            Directory path where patches will be saved
        patch_base_name : str
            Base name for the saved patch files
        patch_size : tuple[int, ...]
            Size of the patch to extract/augment
        axis : Optional[int]
            Axis along which to perform augmentation. If None, augment across all axes.

        Returns
        -------
        tuple[str, str]
            Paths to the saved image and mask files as (image_path, mask_path)
        """
        import os

        from ..utilities.io_util import generate_next_name

        # Augment the image and mask
        augmented_im, augmented_mask = self.augment(im, mask, patch_size, axis)

        # Create subdirectories for input and ground truth
        input_path = os.path.join(patch_path, self.input_dir)
        ground_truth_path = os.path.join(patch_path, self.ground_truth_dir)

        # Ensure the output directories exist
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(ground_truth_path, exist_ok=True)

        # Generate unique filenames
        im_base_name = generate_next_name(
            input_path, f"{patch_base_name}", ".tif"
        )
        mask_base_name = generate_next_name(
            ground_truth_path, f"{patch_base_name}", ".tif"
        )

        im_path = os.path.join(input_path, f"{im_base_name}.tif")
        mask_path = os.path.join(ground_truth_path, f"{mask_base_name}.tif")

        # Save the augmented image and mask
        self._save_array(augmented_im, im_path)
        self._save_array(augmented_mask, mask_path)

        return im_path, mask_path

    def create_valid_coordinates(
        self,
        sparse_annotation: np.ndarray,
        im_shape: tuple[int, ...],
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> list[tuple[int, ...]]:
        """
        Create and cache list of valid crop starting coordinates.

        Parameters
        ----------
        sparse_annotation : np.ndarray
            Sparse annotation array with labeled pixels (non-zero values)
        im_shape : tuple[int, ...]
            Shape of the input image
        patch_size : tuple[int, ...]
            Size of the patch to extract
        axis : Optional[int]
            Specific axis to crop along. If None, crop along all axes.

        Returns
        -------
        list[tuple[int, ...]]
            List of valid starting coordinates
        """
        binary = sparse_annotation != 0
        coords = np.argwhere(binary)

        print("num coords:", len(coords))

        out = np.zeros_like(binary, dtype=bool)

        for z, y, x in coords:
            z0 = max(0, z - patch_size[0] + 1)
            z1 = min(out.shape[0] - patch_size[0] + 1, z + 1)

            y0 = max(0, y - patch_size[1] + 1)
            y1 = min(out.shape[1] - patch_size[1] + 1, y + 1)

            x0 = max(0, x - patch_size[2] + 1)
            x1 = min(out.shape[2] - patch_size[2] + 1, x + 1)

            out[z0:z1, y0:y1, x0:x1] = True

        valid_coords = np.argwhere(out)
        valid_starts = []

        for coord in valid_coords:
            # Check if patch fits within bounds
            if all(  # noqa: SIM102
                c + p <= s
                for c, p, s in zip(coord, patch_size, im_shape, strict=False)
            ):
                # Check axis constraint
                if axis is None or all(  # noqa: SIM102
                    c == 0 if i != axis else True for i, c in enumerate(coord)
                ):
                    valid_starts.append(tuple(coord))

        self.valid_coordinates = valid_starts
        return valid_starts

    def _get_random_crop_indices(
        self,
        im_shape: tuple[int, ...],
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> tuple[int, ...]:
        """
        Generate random starting indices for cropping.

        Parameters
        ----------
        im_shape : tuple[int, ...]
            Shape of the input image
        patch_size : tuple[int, ...]
            Size of the patch to extract
        axis : Optional[int]
            Specific axis to crop along. If None, crop along all axes.

        Returns
        -------
        tuple[int, ...]
            Starting indices for cropping
        """
        # Check if valid coordinates are pre-computed
        if self.valid_coordinates is not None:
            # Use cached valid coordinates
            return self.valid_coordinates[
                np.random.randint(len(self.valid_coordinates))
            ]

        # Standard random cropping
        start_indices = []
        for i, (img_dim, patch_dim) in enumerate(
            zip(im_shape, patch_size, strict=False)
        ):
            if axis is not None and i != axis:
                start_indices.append(0)
            else:
                max_start = img_dim - patch_dim
                start = (
                    np.random.randint(0, max_start + 1) if max_start > 0 else 0
                )
                start_indices.append(start)
        return tuple(start_indices)

    def _save_array(self, array: np.ndarray, filepath: str) -> None:
        """
        Save a numpy array to disk as a TIFF file.

        Parameters
        ----------
        array : np.ndarray
            Array to save
        filepath : str
            Path where the file will be saved
        """
        try:
            import tifffile

            tifffile.imwrite(filepath, array)
        except ImportError:
            # Fallback to numpy if tifffile is not available
            import numpy as np

            np.save(filepath.replace(".tif", ".npy"), array)
