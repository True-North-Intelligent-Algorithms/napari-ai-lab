from dataclasses import dataclass, field

import albumentations as A
import numpy as np

from .augmenter_base import AugmenterBase


@dataclass
class AlbumentationsAugmenter(AugmenterBase):
    """
    An augmenter that uses Albumentations library for image augmentation.

    This augmenter extracts random patches and applies various augmentations
    using the Albumentations library including flips, rotations, crops, and
    brightness/contrast adjustments.
    """

    # Augmenter name
    name: str = field(
        default="AlbumentationsAugmenter", init=False, repr=False
    )

    # Instructions for users
    instructions: str = field(
        default="""
Albumentations Advanced Augmentation:
• Random flips (vertical/horizontal)
• Random 90-degree rotations
• Random sized crop with resize
• Random brightness/contrast adjustments
• Normalization: Percentile-based intensity normalization
• Best for: Advanced data augmentation with diverse transforms
    """,
        init=False,
        repr=False,
    )

    # Augmentation parameters
    normalize: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    use_global_stats: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    do_vertical_flip: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    do_horizontal_flip: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    do_random_rotate90: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    do_random_sized_crop: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": False,
        },
    )

    do_random_brightness_contrast: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
        },
    )

    size_factor: float = field(
        default=0.8,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.1,
            "max": 1.0,
            "step": 0.1,
            "default": 0.8,
        },
    )

    seed: int | None = field(
        default=None,
        metadata={
            "type": "int",
            "param_type": "augmentation",
            "min": 0,
            "max": 99999,
            "step": 1,
            "default": 42,
            "nullable": True,
        },
    )

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__(seed=self.seed)
        self.normalize = self.normalize
        self.use_global_stats = self.use_global_stats
        self.do_vertical_flip = self.do_vertical_flip
        self.do_horizontal_flip = self.do_horizontal_flip
        self.do_random_rotate90 = self.do_random_rotate90
        self.do_random_sized_crop = self.do_random_sized_crop
        self.do_random_brightness_contrast = self.do_random_brightness_contrast
        self.size_factor = self.size_factor

    @classmethod
    def register(cls):
        """Register this augmenter with the framework."""
        return AugmenterBase.register_framework("AlbumentationsAugmenter", cls)

    def get_parameters_dict(self):
        """
        Return current parameter values as a dict (same format as segmenters).
        """
        return {
            "normalize": self.normalize,
            "use_global_stats": self.use_global_stats,
            "do_vertical_flip": self.do_vertical_flip,
            "do_horizontal_flip": self.do_horizontal_flip,
            "do_random_rotate90": self.do_random_rotate90,
            "do_random_sized_crop": self.do_random_sized_crop,
            "do_random_brightness_contrast": self.do_random_brightness_contrast,
            "size_factor": self.size_factor,
            "seed": self.seed,
        }

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
