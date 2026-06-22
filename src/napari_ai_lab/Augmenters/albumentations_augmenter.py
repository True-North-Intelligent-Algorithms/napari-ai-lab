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
        default=True,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": True,
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

    brightness_limit: float = field(
        default=0.9,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.9,
        },
    )

    contrast_limit: float = field(
        default=0.2,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.2,
        },
    )

    normalization_jitter: float = field(
        default=5.0,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 1.0,
            "max": 20.0,
            "step": 0.5,
            "default": 5.0,
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
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self.supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self.normalize = self.normalize
        self.use_global_stats = self.use_global_stats
        self.do_vertical_flip = self.do_vertical_flip
        self.do_horizontal_flip = self.do_horizontal_flip
        self.do_random_rotate90 = self.do_random_rotate90
        self.do_random_sized_crop = self.do_random_sized_crop
        self.do_random_brightness_contrast = self.do_random_brightness_contrast
        self.size_factor = self.size_factor
        self.brightness_limit = self.brightness_limit
        self.contrast_limit = self.contrast_limit
        self.normalization_jitter = self.normalization_jitter

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
            "brightness_limit": self.brightness_limit,
            "contrast_limit": self.contrast_limit,
            "normalization_jitter": self.normalization_jitter,
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
            min_max_height = (
                int(inverse_size_factor * patch_size[0]),
                patch_size[0],
            )
            augmentations.append(
                A.RandomSizedCrop(
                    min_max_height=min_max_height,
                    size=patch_size,
                    p=1.0,
                )
            )

        if self.do_random_brightness_contrast:
            # TODO: add brightness and contrast limits as options
            augmentations.append(
                A.RandomBrightnessContrast(
                    p=0.5,
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                    brightness_by_max=False,
                )
            )

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
        # Apply Albumentations augmentations
        # Assume the last two dimensions are spatial (H, W).
        # For 3D images the transform is applied to each slice along axis 0.

        # Use the 2D patch size (last two dims) for the pipeline
        patch_size_2d = patch_size[-2:] if len(patch_size) >= 2 else patch_size

        transform = self._create_augmentation_pipeline(patch_size_2d)

        # Determine if this is truly a 3D image or just RGB
        # If last dimension is 3, it's RGB (not 3D)
        # If last dimension is > 4, it's probably 3D
        if im.ndim > 2:
            last_dim = im.shape[-1]
            is_3d = last_dim > 4 or last_dim not in [3, 4]
        else:
            is_3d = False

        if not is_3d:
            # Single 2D image — apply transform directly
            augmented = transform(image=im, mask=mask)
            augmented_im = augmented["image"]
            augmented_mask = augmented["mask"]
        else:
            # 3D image — pick a random start along axis 0 so we extract
            # exactly patch_size[0] consecutive slices, then apply the 2D
            # transform independently to each slice.
            patch_z = patch_size[0]
            z_size = im.shape[0]
            if z_size < patch_z:
                raise ValueError(
                    f"Image z-size ({z_size}) is smaller than patch z-size ({patch_z})"
                )
            z_start = np.random.randint(0, z_size - patch_z + 1)
            z_end = z_start + patch_z

            aug_im_slices = []
            aug_mask_slices = []

            replay_transform = A.ReplayCompose(transform.transforms)

            first = replay_transform(image=im[z_start], mask=mask[z_start])

            aug_im_slices.append(first["image"])
            aug_mask_slices.append(first["mask"])

            replay = first["replay"]

            for i in range(z_start + 1, z_end):
                augmented = A.ReplayCompose.replay(
                    replay, image=im[i], mask=mask[i]
                )
                aug_im_slices.append(augmented["image"])
                aug_mask_slices.append(augmented["mask"])

            augmented_im = np.stack(aug_im_slices, axis=0)
            augmented_mask = np.stack(aug_mask_slices, axis=0)

        return augmented_im, augmented_mask
