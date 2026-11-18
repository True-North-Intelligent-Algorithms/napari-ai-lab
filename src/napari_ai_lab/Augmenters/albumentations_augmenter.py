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
• Random sized crop with resize: min/max scale is how much objects are
  resized -- 1.0 keeps their size, 0.8-1.2 shrinks or enlarges up to 20%.
  Off: patches cut at their own size
• min objects: a patch must hold at least this many labelled objects
  (0 allows empty patches); it is redrawn until it does
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

    # How much objects are resized: 1.0 keeps them as they are, 0.5 halves
    # them, 2.0 doubles them. A factor rather than a size in pixels, which
    # needs the objects' size known first; the log says what comes out.
    min_scale: float = field(
        default=0.8,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.1,
            "max": 10.0,
            "step": 0.05,
            "default": 0.8,
        },
    )

    max_scale: float = field(
        default=1.2,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.1,
            "max": 10.0,
            "step": 0.05,
            "default": 1.2,
        },
    )

    # Labelled objects a patch must hold, any part of them; 0 allows empty
    # patches. Redrawn until met, up to MAX_TRIES.
    min_objects: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "augmentation",
            "min": 0,
            "max": 1000,
            "step": 1,
            "default": 1,
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
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "min_objects": self.min_objects,
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
        self,
        patch_size: int,
        min_max_height: tuple[int, int] | None = None,
        bbox_params: A.BboxParams | None = None,
    ) -> A.Compose:
        """
        Create the Albumentations augmentation pipeline.

        Parameters
        ----------
        patch_size : int
            Size of the patch (assumes square patches)
        bbox_params : A.BboxParams, optional
            Set when the pipeline carries boxes as well as the image, so
            every transform moves them too.

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

        if min_max_height is not None:
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

        return A.Compose(augmentations, bbox_params=bbox_params)

    def _report_once(self, message: str) -> None:
        """Print a message the first time only. Patch generation runs hundreds
        of times over the same images, and a warning repeated hundreds of times
        is a warning nobody reads. Kept, in order, for the Augment tab's log
        (``take_reports``)."""
        seen = self.__dict__.setdefault("_reported", {})
        if message not in seen:
            seen[message] = None
            print(f"  {message}")

    def take_reports(self) -> list[str]:
        """The messages reported since last asked, oldest first; cleared, so
        the next run says them again."""
        return list(self.__dict__.pop("_reported", {}))

    #: Draws of a patch before giving up on min_objects.
    MAX_TRIES = 30

    def _crop_size(self, patch_size, image_shape, longest=None):
        """The square crop to cut, for the patch it is resized to.

        Cropping *small* enlarges what is in it, cropping *large* shrinks it:
        a scale s means a crop of ``patch / s``. The scale is drawn
        log-uniformly between min_scale and max_scale -- each doubling as
        likely -- or is 1 with random sized crop off. Clamped to the image,
        which a crop cannot exceed. *longest*, the largest object's long
        axis in pixels, is only for saying what comes out.
        """
        lo, hi = sorted((self.min_scale, self.max_scale))
        lo = max(lo, 1e-3)
        if not self.do_random_sized_crop:
            lo = hi = 1.0
        side = patch_size[0]
        limit = min(image_shape[-2:])
        if longest:
            self._report_once(
                f"largest object {longest:.0f}px -> {longest * lo:.0f}-"
                f"{longest * hi:.0f}px in {side}px patches (scale {lo:g}-{hi:g})"
            )
        if side / lo > limit:
            self._report_once(
                f"scale: this image is only {limit}px, so objects shrink to "
                f"{side / limit:.2f} at most, not {lo:g}"
            )
        scale = np.exp(self.rng.uniform(np.log(lo), np.log(hi)))
        size = int(round(min(max(side / scale, 16), limit)))
        return (size, size)

    @staticmethod
    def _longest(mask):
        """Long axis of the largest labelled object in *mask*, or None."""
        from skimage.measure import regionprops

        labels = mask if mask.ndim == 2 else mask[mask.shape[0] // 2]
        lengths = [
            r.axis_major_length for r in regionprops(labels.astype(np.int32))
        ]
        return max(lengths) if lengths else None

    @staticmethod
    def _count_objects(mask) -> int:
        """Distinct labelled objects in *mask*, any part of them."""
        return int(np.count_nonzero(np.unique(mask)))

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
        longest = self._longest(mask)
        available = self._count_objects(mask)
        wanted = min(self.min_objects, available)
        if available < self.min_objects:
            self._report_once(
                f"min objects: asked for {self.min_objects} per patch, but "
                f"this image has only {available}"
            )

        best = None
        for _ in range(self.MAX_TRIES if wanted > 0 else 1):
            out = self._augment_once(
                im, mask, patch_size, patch_size_2d, longest
            )
            n = self._count_objects(out[1])
            if best is None or n > best[0]:
                best = (n, out)
            if n >= wanted:
                break
        if best[0] < wanted:
            self._report_once(
                f"min objects: some patches could not get {wanted} objects in "
                f"{self.MAX_TRIES} tries and kept the most found. Raise the "
                f"patch size or lower min objects"
            )
        return best[1]

    def _augment_once(self, im, mask, patch_size, patch_size_2d, longest):
        """One augmented patch of *im* and *mask*; see ``augment``."""
        transform = self._create_augmentation_pipeline(
            patch_size_2d,
            self._crop_size(patch_size_2d, mask.shape, longest),
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

    def augment_boxes(
        self,
        im: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        patch_size: tuple[int, ...],
        min_visibility: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment a 2D image and its boxes together, for YOLO training.

        The same flips, rotations, crop and colour changes as ``augment``,
        with each box moved along with the image. A box mostly cut off by
        the crop -- less than *min_visibility* of it left -- is dropped, and
        its class with it.

        An image with no boxes is still augmented, and comes back with none:
        a background example, which YOLO learns "nothing here" from.

        Parameters
        ----------
        im : np.ndarray
            A YX or YXC image.
        boxes : np.ndarray
            (n, 4) boxes in YOLO form: centre x, centre y, width, height,
            each as a fraction of the image size.
        classes : np.ndarray
            (n,) class id of each box.
        patch_size : tuple[int, ...]
            Size of the patch; the last two values are used.
        min_visibility : float
            Fraction of a box that must survive the crop for it to be kept.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The augmented image, its boxes in YOLO form, and their classes.
        """
        is_rgb = im.ndim == 3 and im.shape[-1] in (3, 4)
        if im.ndim != 2 and not is_rgb:
            raise ValueError(
                f"Box augmentation is 2D only (YX or YXC); got shape {im.shape}"
            )

        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        classes = np.asarray(classes, dtype=int).reshape(-1)
        patch_size_2d = tuple(patch_size[-2:])

        # The same scale range as augment; the largest box only says what
        # comes out.
        height, width = im.shape[:2]
        longest = None
        if len(boxes):
            longest = max(
                np.max(boxes[:, 2]) * width, np.max(boxes[:, 3]) * height
            )
        crop = self._crop_size(patch_size_2d, im.shape[:2], longest)

        transform = self._create_augmentation_pipeline(
            patch_size_2d,
            crop,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["classes"],
                min_visibility=min_visibility,
                # Rounding in a saved txt can put an edge at 1.000001.
                clip=True,
            ),
        )

        # See augment: ColorJitter takes only uint8 and float32.
        if self.do_color_jitter and im.dtype not in (np.uint8, np.float32):
            im = im.astype(np.float32)

        augmented = transform(image=im, bboxes=boxes, classes=classes)
        out_boxes = np.asarray(augmented["bboxes"], dtype=float).reshape(-1, 4)
        out_classes = np.asarray(augmented["classes"], dtype=int).reshape(-1)
        return augmented["image"], out_boxes, out_classes
