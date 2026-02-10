import numpy as np

from .augmenter_base import AugmenterBase


class SimpleAugmenter(AugmenterBase):
    """
    A simple augmenter that performs random cropping of images and masks.

    This augmenter extracts random patches of specified size from the input
    image and mask along the specified axis.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the SimpleAugmenter.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility. If None, uses random state.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def augment(
        self,
        im: np.ndarray,
        mask: np.ndarray,
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment an image and mask by performing random cropping.

        Parameters
        ----------
        im : np.ndarray
            Input image array
        mask : np.ndarray
            Input mask array corresponding to the image
        patch_size : tuple[int, ...]
            Size of the patch to extract
        axis : Optional[int]
            Axis along which to perform cropping. If None, crop across all axes.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Randomly cropped image and mask as a tuple (cropped_image, cropped_mask)

        Raises
        ------
        ValueError
            If patch_size dimensions don't match image dimensions or if patch is larger than image
        """
        if im.shape != mask.shape:
            raise ValueError(
                f"Image and mask shapes must match. Got image: {im.shape}, mask: {mask.shape}"
            )

        # Convert patch_size to tuple if it's a single value
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * im.ndim

        if len(patch_size) != im.ndim:
            raise ValueError(
                f"patch_size dimensions ({len(patch_size)}) must match "
                f"image dimensions ({im.ndim})"
            )

        # Check if patch size is valid
        for i, (img_dim, patch_dim) in enumerate(
            zip(im.shape, patch_size, strict=False)
        ):
            if patch_dim > img_dim:
                raise ValueError(
                    f"Patch size ({patch_dim}) at dimension {i} is larger than "
                    f"image size ({img_dim})"
                )

        # Generate random starting indices for cropping
        start_indices = self._get_random_crop_indices(
            im.shape, patch_size, axis
        )

        # Create slicing tuples for cropping
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(start_indices, patch_size, strict=False)
        )

        # Crop both image and mask using the same indices
        cropped_im = im[slices]
        cropped_mask = mask[slices]

        return cropped_im, cropped_mask

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
        start_indices = []

        for i, (img_dim, patch_dim) in enumerate(
            zip(im_shape, patch_size, strict=False)
        ):
            if axis is not None and i != axis:
                # If axis is specified and this is not the axis, start at 0
                # and use the full dimension
                start_indices.append(0)
            else:
                # Random start position ensuring patch fits within image
                max_start = img_dim - patch_dim
                start = (
                    np.random.randint(0, max_start + 1) if max_start > 0 else 0
                )
                start_indices.append(start)

        return tuple(start_indices)
