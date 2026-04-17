from dataclasses import dataclass, field

import numpy as np

from .augmenter_base import AugmenterBase


@dataclass
class SimpleAugmenter(AugmenterBase):
    """
    A simple augmenter that performs random cropping of images and masks.

    This augmenter extracts random patches of specified size from the input
    image and mask along the specified axis.
    """

    # Augmenter name
    name: str = field(default="SimpleAugmenter", init=False, repr=False)

    # Instructions for users
    instructions: str = field(
        default="""
Simple Random Crop Augmentation:
• Randomly crops patches from images and annotations
• Normalization: Percentile-based intensity normalization
• Use Global Stats: Apply same normalization across all patches
• Seed: Set for reproducible augmentation (optional)
• Best for: Basic data augmentation for training
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
        self._potential_axes = ["YX", "ZYX"]
        self.supported_axes = ["YX", "ZYX"]
        self.normalize = self.normalize
        self.use_global_stats = self.use_global_stats

    @classmethod
    def register(cls):
        """Register this augmenter with the framework."""
        return AugmenterBase.register_framework("SimpleAugmenter", cls)

    def get_parameters_dict(self):
        """
        Return current parameter values as a dict (same format as segmenters).
        """
        return {
            "normalize": self.normalize,
            "use_global_stats": self.use_global_stats,
            "seed": self.seed,
        }

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

        # Normalize image if enabled
        if self.normalize:
            cropped_im = self.normalize_image(
                cropped_im, use_global_stats=self.use_global_stats
            )

        return cropped_im, cropped_mask
