"""
scikit-image Watershed Global Segmenter.

Classical segmentation pipeline built on top of ``scikit-image``:

    gaussian blur → threshold → remove small holes → remove small
    objects → distance transform → local-maxima seeds → watershed →
    connected-component labels

Follows the same dataclass/registry pattern as ``StardistSegmenter`` and
``CellposeSegmenter`` so the widget UI (``NDOperationWidget``) picks up
its parameters automatically.
"""

from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Optional dependency: scikit-image + scipy
try:
    from scipy import ndimage as ndi
    from skimage import filters, measure, morphology, segmentation
    from skimage.feature import peak_local_max

    _is_skimage_available = True
except ImportError:  # pragma: no cover - only exercised without deps
    ndi = None
    filters = None
    measure = None
    morphology = None
    segmentation = None
    peak_local_max = None
    _is_skimage_available = False


@dataclass
class SkImageWatershedSegmenter(GlobalSegmenterBase):
    """
    Classical watershed segmenter using scikit-image.

    Pipeline:
      1. Gaussian blur (``sigma``)
      2. Threshold on normalized image (``threshold``)
      3. Fill small holes (``min_hole_size``)
      4. Remove small objects (``min_obj_size``)
      5. Distance transform on binary mask
      6. Local-maxima seeds (``peak_min_distance``)
      7. Watershed → labeled instances
    """

    instructions = """
scikit-image Watershed Segmentation:
• Classical pipeline: blur → threshold → clean → distance → watershed
• sigma: Gaussian blur strength (higher = smoother)
• threshold: Foreground cutoff on the normalized image (0–1)
• min_hole_size: Fill holes smaller than this many pixels
• min_obj_size: Remove foreground blobs smaller than this many pixels
• peak_min_distance: Minimum distance between watershed seeds
• invert: Segment dark objects on a bright background
    """

    sigma: float = field(
        default=1.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "default": 1.0,
        },
    )

    threshold: float = field(
        default=0.3,
        metadata={
            "type": "float",
            "param_type": "inference",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.3,
        },
    )

    min_hole_size: int = field(
        default=64,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 10000,
            "step": 8,
            "default": 64,
        },
    )

    min_obj_size: int = field(
        default=64,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 10000,
            "step": 8,
            "default": 64,
        },
    )

    peak_min_distance: int = field(
        default=10,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 1,
            "max": 200,
            "step": 1,
            "default": 10,
        },
    )

    invert: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "default": False,
        },
    )

    def __post_init__(self):
        """Initialize base state after dataclass init."""
        super().__init__()
        self._supported_axes = ["YX", "ZYX"]
        self._potential_axes = ["YX", "ZYX"]

    def are_dependencies_available(self):
        """Return True if scikit-image + scipy are importable."""
        return _is_skimage_available

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Collapse multi-channel images to grayscale and normalize to [0,1]."""
        arr = image
        # Collapse a trailing channel axis of size 3/4 to grayscale.
        if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
            if arr.shape[-1] == 3:
                arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                arr = np.mean(arr, axis=-1)

        arr = arr.astype(np.float32, copy=False)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        if self.invert:
            arr = 1.0 - arr
        return arr

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def segment(self, image, **kwargs):
        """
        Run the watershed pipeline on ``image`` and return an int label map.
        """
        if not _is_skimage_available:
            raise RuntimeError(
                "SkImageWatershedSegmenter requires scikit-image and scipy."
            )

        if image.ndim < 2:
            raise ValueError(
                f"SkImageWatershedSegmenter needs 2D or 3D input. Got shape {image.shape}"
            )

        norm = self._prepare_image(image)

        # 1. Blur
        blur = filters.gaussian(norm, sigma=self.sigma)

        # 2. Threshold
        binary = blur >= self.threshold

        # 3/4. Morphological cleanup
        if self.min_hole_size > 0:
            binary = morphology.remove_small_holes(
                binary, area_threshold=self.min_hole_size
            )
        if self.min_obj_size > 0:
            binary = morphology.remove_small_objects(
                binary, min_size=self.min_obj_size
            )

        # If nothing survived cleanup, return an empty label map early.
        if not binary.any():
            print("SkImageWatershed: no foreground pixels after cleanup.")
            return np.zeros(binary.shape, dtype=np.uint16)

        # 5. Distance transform
        distance = ndi.distance_transform_edt(binary)

        # 6. Seed points from local maxima
        coords = peak_local_max(
            distance,
            min_distance=max(1, int(self.peak_min_distance)),
            labels=binary,
        )
        markers_array = np.zeros(distance.shape, dtype=bool)
        if coords.size:
            markers_array[tuple(coords.T)] = True
        markers, _ = ndi.label(markers_array)

        # 7. Watershed
        labels = segmentation.watershed(-distance, markers, mask=binary)

        num_objects = int(labels.max())
        print(
            f"SkImageWatershed: {num_objects} objects "
            f"(sigma={self.sigma}, threshold={self.threshold}, "
            f"peak_min_distance={self.peak_min_distance})"
        )
        return labels.astype(np.uint16)

    def get_parameters_dict(self):
        """Return current inference parameters as a dict."""
        return {
            "sigma": self.sigma,
            "threshold": self.threshold,
            "min_hole_size": self.min_hole_size,
            "min_obj_size": self.min_obj_size,
            "peak_min_distance": self.peak_min_distance,
            "invert": self.invert,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the global segmenter framework."""
        return GlobalSegmenterBase.register_framework(
            "SkImageWatershedSegmenter", cls
        )
