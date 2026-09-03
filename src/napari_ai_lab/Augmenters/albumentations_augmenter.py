# Keeps annotations as strings rather than evaluating them at import time.
# Without it, `-> A.Compose` below is evaluated when the class body runs, which
# is None.Compose on a machine without albumentations -- an import-time crash
# the try/except cannot prevent.
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import albumentations as A
except ImportError:  # optional dependency
    A = None

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
• Colour jitter (hue/saturation) and random gamma, off by default
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

    do_color_jitter: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": False,
        },
    )

    do_random_gamma: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "augmentation",
            "default": False,
        },
    )

    min_long_axis: int = field(
        default=30,
        metadata={
            "type": "int",
            "param_type": "augmentation",
            "min": 4,
            "max": 2048,
            "step": 5,
            "default": 30,
        },
    )

    max_long_axis: int = field(
        default=200,
        metadata={
            "type": "int",
            "param_type": "augmentation",
            "min": 8,
            "max": 4096,
            "step": 5,
            "default": 200,
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

    hue_limit: float = field(
        default=0.05,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.0,
            "max": 0.5,
            "step": 0.01,
            "default": 0.05,
        },
    )

    saturation_limit: float = field(
        default=0.3,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.3,
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
        """Initialize parent class after dataclass initialization.

        Constructs even when albumentations is missing, so the augmenter can
        appear in the list with a red banner rather than not appearing at all.
        The guard lives in ``_create_augmentation_pipeline``, at the point the
        library is actually needed -- the same shape as CellposeSegmenter,
        which constructs without cellpose and checks at run time. See
        docs/spec/0003-optional-dependencies.md.
        """
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
        self.min_long_axis = self.min_long_axis
        self.max_long_axis = self.max_long_axis
        self.brightness_limit = self.brightness_limit
        self.contrast_limit = self.contrast_limit
        self.normalization_jitter = self.normalization_jitter

    @classmethod
    def are_dependencies_available(cls) -> bool:
        """False when albumentations is not installed, so the banner goes red."""
        return A is not None

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
            "min_long_axis": self.min_long_axis,
            "max_long_axis": self.max_long_axis,
            "do_color_jitter": self.do_color_jitter,
            "do_random_gamma": self.do_random_gamma,
            "brightness_limit": self.brightness_limit,
            "contrast_limit": self.contrast_limit,
            "hue_limit": self.hue_limit,
            "saturation_limit": self.saturation_limit,
            "normalization_jitter": self.normalization_jitter,
            "seed": self.seed,
        }

    def _create_augmentation_pipeline(
        self, patch_size: int, min_max_height: tuple[int, int] | None = None
    ) -> A.Compose:
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

        Raises
        ------
        ImportError
            If albumentations is not installed. It is an optional dependency;
            SimpleAugmenter works without it. Raised here rather than at
            construction so the augmenter still lists with a red banner.
        """
        if A is None:
            raise ImportError(
                "AlbumentationsAugmenter requires albumentations, which is "
                "not installed. Install it with: pip install albumentations"
            )

        augmentations = []

        if self.do_vertical_flip:
            augmentations.append(A.VerticalFlip(p=0.5))

        if self.do_horizontal_flip:
            augmentations.append(A.HorizontalFlip(p=0.5))

        if self.do_random_rotate90:
            augmentations.append(A.RandomRotate90(p=0.5))

        if self.do_random_sized_crop and min_max_height is not None:
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

        # For RGB, where hue and saturation vary between acquisitions.
        # Off by default: single-channel images have no hue to shift.
        if self.do_color_jitter:
            augmentations.append(
                # Hue and saturation only -- brightness and contrast have
                # their own toggle above, and ColorJitter reads its limits
                # multiplicatively where RandomBrightnessContrast reads them
                # additively.
                # The shift is drawn uniformly up to the limit, so hue_limit
                # sets how far colour can move, not how far it does. At the
                # 0.5 maximum a third of patches are untouched, the rest are
                # spread over the whole colour circle.
                A.ColorJitter(
                    hue=self.hue_limit,
                    saturation=self.saturation_limit,
                    brightness=0.0,
                    contrast=0.0,
                    p=0.65,
                )
            )

        if self.do_random_gamma:
            augmentations.append(A.RandomGamma(p=0.5))

        return A.Compose(augmentations)

    def _report_once(self, message: str) -> None:
        """Print a message the first time only. Patch generation runs hundreds
        of times over the same images, and a warning repeated hundreds of times
        is a warning nobody reads."""
        seen = self.__dict__.setdefault("_reported", set())
        if message not in seen:
            seen.add(message)
            print(f"  {message}")

    def _crop_range(self, mask, patch_size) -> tuple[int, int] | None:
        """The crop sizes that put objects inside the requested size range.

        A crop is resized to the patch, so cropping *small* enlarges what is in
        it and cropping *large* shrinks it. To land an object of long axis L at
        a target size T, crop ``patch_size * L / T``.

        Anchored on the **largest** labelled object, not the median. The point
        of a range is that the top of it gets covered, and the biggest object
        is what has to reach it -- a median anchor leaves the large end empty
        unless the labels happen to be uniform. Objects smaller than the
        requested minimum simply come out smaller still, which costs nothing:
        extra small examples are harmless, missing large ones are not.

        Returns None when there is nothing to measure, which leaves the crop
        out of the pipeline rather than guessing a scale.
        """
        from skimage.measure import regionprops

        labels = mask if mask.ndim == 2 else mask[mask.shape[0] // 2]
        lengths = [
            r.axis_major_length for r in regionprops(labels.astype(np.int32))
        ]
        if not lengths:
            return None

        longest = max(lengths)
        if longest <= 0:
            return None

        # crop = patch * longest / target, so the smaller target gives the
        # larger crop. Clamped so a crop is never below 16px or above the
        # image, which would fail rather than scale.
        low = int(round(patch_size[0] * longest / max(self.max_long_axis, 1)))
        high = int(round(patch_size[0] * longest / max(self.min_long_axis, 1)))
        # Shrinking an object means cropping *more* than the patch, and there
        # is only so much image. Say so rather than quietly narrowing the
        # range: a request that cannot be met is worth knowing before training,
        # not after wondering why the small end is empty.
        limit = min(mask.shape[-2:])
        if high > limit:
            reachable = int(round(longest * patch_size[0] / limit))
            self._report_once(
                f"size range: asked for {self.min_long_axis}-"
                f"{self.max_long_axis}px long axis, but this image can only "
                f"reach {reachable}-{self.max_long_axis}px -- shrinking "
                f"further would need a crop larger than the image"
            )
        if low < patch_size[0] // 2:
            self._report_once(
                f"size range: {self.max_long_axis}px objects are more than "
                f"half of a {patch_size[0]}px patch, so most will be cut by "
                f"the crop edge -- raise the patch size or lower max_long_axis"
            )
        low, high = max(16, min(low, limit)), max(16, min(high, limit))
        if low > high:
            low, high = high, low

        # One size per call, drawn log-uniformly, rather than handing
        # RandomSizedCrop the whole range. It samples the crop *height*
        # uniformly, and size is the reciprocal of height -- so a uniform
        # height draw piles up at the small-object end and the top of the
        # requested range goes nearly unvisited. Log-uniform gives each
        # doubling equal weight, which is what "cover 30 to 200" means.
        draw = np.exp(self.rng.uniform(np.log(low), np.log(high)))
        size = int(round(draw))
        return (size, size)

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

        transform = self._create_augmentation_pipeline(
            patch_size_2d, self._crop_range(mask, patch_size_2d)
        )

        # Determine if this is truly a 3D image or just RGB
        # If last dimension is 3, it's RGB (not 3D)
        # ColorJitter converts RGB to HSV through OpenCV, which accepts uint8
        # and float32 and nothing else -- float64 and uint16 both raise
        # "Unsupported depth of input image". Normalisation usually leaves
        # float32, but it can be switched off, so convert here rather than
        # relying on it.
        if self.do_color_jitter and im.dtype not in (np.uint8, np.float32):
            im = im.astype(np.float32)

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
