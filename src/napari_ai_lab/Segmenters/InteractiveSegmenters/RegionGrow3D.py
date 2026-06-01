"""
RegionGrow3D Interactive Segmenter.

Simple, fast seeded 3D region growing (flood fill) intended as an
annotation-assistance tool — not a deep-learning model.

For each positive seed point the segmenter:
  1. Extracts a local cube of size ``window_size`` centred on the seed.
  2. (Optionally) Gaussian-smooths the cube.
  3. Builds a candidate binary mask using asymmetric intensity bounds
     ``[I_seed - tolerance_below, I_seed + tolerance_above]`` and then runs
     :func:`skimage.segmentation.flood` on that binary mask from the seed
     voxel with the configured ``connectivity`` to pick the connected
     component containing the seed.
  4. Pastes the resulting binary mask back into a full-sized output volume.

Results from multiple seeds are combined by union.  Shapes are ignored
(matching how SAM3D treats them as optional context only).
"""

from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import flood

    _is_available = True
except ImportError:
    flood = None
    gaussian_filter = None
    _is_available = False
    print(
        "Warning: scikit-image / scipy not installed. RegionGrow3D will not work."
    )


@dataclass
class RegionGrow3D(InteractiveSegmenterBase):
    """
    Local seeded 3D region-growing segmenter.

    Click one or more points; each becomes a seed for a flood fill
    restricted to a local cube around it.  Returns a binary uint8 mask
    the same shape as the input volume.
    """

    # Opt-in: the host app may re-run segment() with the last seed points
    # whenever any harvested parameter changes, giving live visual feedback.
    supports_live_param_update = True

    instructions = """
Instructions for Region Grow 3D:
1. Click one or more points inside the structure of interest.
2. A local cube of side `window_size` is taken around each seed.
3. Voxels are grown while (I_seed - tolerance_below) <= I <= (I_seed + tolerance_above).
   Use asymmetric bounds to grow more toward brighter or dimmer pixels.
4. `connectivity` chooses 6 / 18 / 26 neighbours.
5. Optional Gaussian smoothing (`gaussian_sigma`) for noisy data.
6. Multiple positive points are combined by union.
    """

    window_size: int = field(
        default=64,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 4,
            "max": 512,
            "step": 2,
            "default": 64,
        },
    )

    tolerance_below: float = field(
        default=100,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1e6,
            "step": 10.0,
            "default": 100,
        },
    )

    tolerance_above: float = field(
        default=100.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1e6,
            "step": 10.0,
            "default": 1000,
        },
    )

    connectivity: int = field(
        default=6,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "choices": [6, 18, 26],
            "default": 6,
        },
    )

    gaussian_sigma: float = field(
        default=0.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "default": 0.0,
        },
    )

    def __init__(self):
        """Initialize the RegionGrow3D segmenter."""
        super().__init__()
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX", "ZYXC"]
        self.selected_axis = self._supported_axes[0]
        # Optional bbox of the most recent segmentation (tuple of slices) or
        # None if no seeds were processed.  Mirrors Otsu3D.last_roi_bbox.
        self.last_roi_bbox = None

    @property
    def supported_axes(self):
        """Get the list of axis configurations this segmenter supports."""
        return self._supported_axes

    @supported_axes.setter
    def supported_axes(self, value):
        self._supported_axes = value

    @property
    def potential_axes(self):
        """Get the list of all axis configurations this could potentially support."""
        return self._potential_axes

    @potential_axes.setter
    def potential_axes(self, value):
        self._potential_axes = value

    def are_dependencies_available(self):
        """Check if required dependencies (scikit-image, scipy) are available."""
        return _is_available

    # ------------------------------------------------------------------
    # Connectivity helper
    # ------------------------------------------------------------------
    @staticmethod
    def _connectivity_rank(connectivity: int) -> int:
        """Map 6/18/26 neighbour counts to skimage's connectivity rank (1/2/3)."""
        if connectivity <= 6:
            return 1
        if connectivity <= 18:
            return 2
        return 3

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform seeded 3D region growing.

        Args:
            image (numpy.ndarray): 3D (ZYX) or 4D (ZYXC) input volume.
            points (list, optional): Seed points. Last 3 components are (Z, Y, X).
            shapes (list, optional): Ignored (kept for API compatibility).
            **kwargs: Unused.

        Returns:
            numpy.ndarray: uint8 binary mask, same spatial shape as input.
        """
        if not _is_available:
            print("RegionGrow3D: dependencies unavailable, returning None")
            return None

        if image.ndim not in (3, 4):
            raise ValueError(
                f"RegionGrow3D requires 3D image. Got shape: {image.shape}"
            )

        # Collapse multichannel (ZYXC) to grayscale (ZYX) the same way Otsu3D does.
        if image.ndim == 4:
            if image.shape[-1] == 3:
                image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                image_gray = np.mean(image, axis=-1)
        else:
            image_gray = image

        d, h, w = image_gray.shape[:3]
        full_mask = np.zeros((d, h, w), dtype=np.uint8)

        if points is None or len(points) == 0:
            print("RegionGrow3D: no seed points provided")
            self.last_roi_bbox = None
            return full_mask

        half = max(1, int(self.window_size) // 2)
        rank = self._connectivity_rank(int(self.connectivity))

        # Track aggregate bbox of all ROIs touched (for last_roi_bbox).
        z_lo = y_lo = x_lo = None
        z_hi = y_hi = x_hi = None

        for raw_pt in points:
            pt = np.asarray(raw_pt)
            if pt.size < 3:
                # 2D point — skip (RegionGrow3D needs a Z coordinate).
                continue
            cz = int(round(float(pt[-3])))
            cy = int(round(float(pt[-2])))
            cx = int(round(float(pt[-1])))

            if not (0 <= cz < d and 0 <= cy < h and 0 <= cx < w):
                print(
                    f"RegionGrow3D: seed ({cz},{cy},{cx}) outside volume — skipped"
                )
                continue

            z0 = max(0, cz - half)
            z1 = min(d, cz + half)
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)

            cube = image_gray[z0:z1, y0:y1, x0:x1]
            if cube.size == 0:
                continue

            cube_f = cube.astype(np.float32, copy=False)
            if self.gaussian_sigma > 0:
                cube_f = gaussian_filter(
                    cube_f, sigma=float(self.gaussian_sigma)
                )

            seed_local = (cz - z0, cy - y0, cx - x0)
            seed_val = float(cube_f[seed_local])
            lo = seed_val - float(self.tolerance_below)
            hi = seed_val + float(self.tolerance_above)
            candidate = (cube_f >= lo) & (cube_f <= hi)
            if not candidate[seed_local]:
                # Numerical edge case — force seed in so flood has a start.
                candidate[seed_local] = True
            try:
                grown = flood(
                    candidate,
                    seed_point=seed_local,
                    connectivity=rank,
                )
            except (ValueError, IndexError) as e:
                print(f"RegionGrow3D: flood failed at {seed_local}: {e}")
                continue

            # Union into the full mask (this cube only).
            sub = full_mask[z0:z1, y0:y1, x0:x1]
            np.logical_or(sub.astype(bool), grown, out=grown)
            full_mask[z0:z1, y0:y1, x0:x1] = grown.astype(np.uint8)

            # Update aggregate bbox.
            z_lo = z0 if z_lo is None else min(z_lo, z0)
            y_lo = y0 if y_lo is None else min(y_lo, y0)
            x_lo = x0 if x_lo is None else min(x_lo, x0)
            z_hi = z1 if z_hi is None else max(z_hi, z1)
            y_hi = y1 if y_hi is None else max(y_hi, y1)
            x_hi = x1 if x_hi is None else max(x_hi, x1)

            print(
                f"RegionGrow3D: seed=({cz},{cy},{cx}) "
                f"cube z=[{z0}:{z1}] y=[{y0}:{y1}] x=[{x0}:{x1}] "
                f"voxels_grown={int(grown.sum())}"
            )

        if z_lo is None:
            self.last_roi_bbox = None
        else:
            self.last_roi_bbox = (
                slice(z_lo, z_hi),
                slice(y_lo, y_hi),
                slice(x_lo, x_hi),
            )

        return full_mask

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("RegionGrow3D", cls)
