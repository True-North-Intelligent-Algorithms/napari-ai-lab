import albumentations as A
import numpy as np

from .augmenter_base import AugmenterBase


class AlbumentationsAugmenter(AugmenterBase):
    """
    An augmenter that uses Albumentations library for image augmentation.

    This augmenter extracts random patches and applies various augmentations
    using the Albumentations library including flips, rotations, crops, and
    brightness/contrast adjustments.
    """

    def __init__(
        self,
        seed: int | None = None,
        normalize: bool = True,
        use_global_stats: bool = True,
        do_vertical_flip: bool = True,
        do_horizontal_flip: bool = True,
        do_random_rotate90: bool = True,
        do_random_sized_crop: bool = False,
        do_random_brightness_contrast: bool = True,
        size_factor: float = 0.8,
    ):
        """
        Initialize the AlbumentationsAugmenter.

        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility. If None, uses random state.
        normalize : bool
            Whether to normalize images using percentile normalization. Default is True.
        use_global_stats : bool
            If True, use global normalization statistics (computed from full image).
            If False, compute normalization statistics from each patch individually.
            Default is True for consistency with inference normalization.
        do_vertical_flip : bool
            Whether to apply vertical flips. Default is True.
        do_horizontal_flip : bool
            Whether to apply horizontal flips. Default is True.
        do_random_rotate90 : bool
            Whether to apply random 90-degree rotations. Default is True.
        do_random_sized_crop : bool
            Whether to apply random sized crop (with resize). Default is False.
        do_random_brightness_contrast : bool
            Whether to apply random brightness/contrast adjustments. Default is True.
        size_factor : float
            Size factor for random sized crop (0.0 to 1.0). Default is 0.8.
        """
        super().__init__()  # Initialize parent class to set up directories
        self.seed = seed
        self.normalize = normalize
        self.use_global_stats = use_global_stats
        self.do_vertical_flip = do_vertical_flip
        self.do_horizontal_flip = do_horizontal_flip
        self.do_random_rotate90 = do_random_rotate90
        self.do_random_sized_crop = do_random_sized_crop
        self.do_random_brightness_contrast = do_random_brightness_contrast
        self.size_factor = size_factor

        if seed is not None:
            np.random.seed(seed)

    def _create_augmentation_pipeline(self, patch_size: int) -> A.Compose:
        """
        Create the Albumentations augmentation pipeline.

        Parameters
        ----------
        patch_size : int
            Size of the patch (assumes square patches)

        Returns
        -------
        A.Compose
            Composed augmentation pipeline
        """
        augmentations = []

        if self.do_vertical_flip:
            augmentations.append(A.VerticalFlip(p=0.5))

        if self.do_horizontal_flip:
            augmentations.append(A.HorizontalFlip(p=0.5))

        if self.do_random_rotate90:
            augmentations.append(A.RandomRotate90(p=0.5))

        if self.do_random_sized_crop:
            # TODO: make more flexibility for resize
            # need to invert the size factor because it controls the crop size which is then resized to the patch size.
            # So a smaller factor will lead to a larger resize.
            inverse_size_factor = 0.99 / self.size_factor
            min_max_height = int(inverse_size_factor * patch_size), patch_size
            augmentations.append(
                A.RandomSizedCrop(
                    min_max_height=min_max_height,
                    size=(patch_size, patch_size),
                    p=0.5,
                )
            )

        if self.do_random_brightness_contrast:
            # TODO: add brightness and contrast limits as options
            augmentations.append(A.RandomBrightnessContrast(p=0.8))

        return A.Compose(augmentations)

    def augment(
        self,
        im: np.ndarray,
        mask: np.ndarray,
        patch_size: tuple[int, ...],
        axis: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Augment an image and mask by performing random cropping and augmentations.

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
            Augmented image and mask as a tuple (augmented_image, augmented_mask)

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

        # Normalize image if enabled (before augmentation)
        if self.normalize:
            cropped_im = self.normalize_image(
                cropped_im, use_global_stats=self.use_global_stats
            )

        # Apply Albumentations augmentations
        # Assume the last two dimensions are spatial (H, W)
        if cropped_im.ndim == 2:
            # 2D image
            patch_size_2d = cropped_im.shape[0]  # Assume square
            transform = self._create_augmentation_pipeline(patch_size_2d)

            augmented = transform(image=cropped_im, mask=cropped_mask)
            augmented_im = augmented["image"]
            augmented_mask = augmented["mask"]
        elif cropped_im.ndim == 3:
            # 3D image - apply augmentation to 2D slice
            # Assume shape is (Z, Y, X) and we want to augment (Y, X)
            patch_size_2d = cropped_im.shape[1]  # Assume square in YX
            transform = self._create_augmentation_pipeline(patch_size_2d)

            # Apply to the 2D slice (assuming first dimension is Z and only 1 slice)
            if cropped_im.shape[0] == 1:
                augmented = transform(
                    image=cropped_im[0], mask=cropped_mask[0]
                )
                augmented_im = augmented["image"][np.newaxis, ...]
                augmented_mask = augmented["mask"][np.newaxis, ...]
            else:
                # For multi-slice, just return cropped (no augmentation)
                augmented_im = cropped_im
                augmented_mask = cropped_mask
        else:
            raise ValueError(
                f"Unsupported image dimensions: {cropped_im.ndim}"
            )

        return augmented_im, augmented_mask
