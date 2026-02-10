from abc import ABC, abstractmethod

import numpy as np


class AugmenterBase(ABC):
    """
    Abstract base class for image and mask augmentation.

    This class provides the interface for augmenting images and their corresponding
    masks, with support for saving augmented patches to disk.
    """

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

        # Ensure the output directory exists
        os.makedirs(patch_path, exist_ok=True)

        # Generate unique filenames
        im_base_name = generate_next_name(
            patch_path, f"{patch_base_name}_im", ".tif"
        )
        mask_base_name = generate_next_name(
            patch_path, f"{patch_base_name}_mask", ".tif"
        )

        im_path = os.path.join(patch_path, f"{im_base_name}.tif")
        mask_path = os.path.join(patch_path, f"{mask_base_name}.tif")

        # Save the augmented image and mask
        self._save_array(augmented_im, im_path)
        self._save_array(augmented_mask, mask_path)

        return im_path, mask_path

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
