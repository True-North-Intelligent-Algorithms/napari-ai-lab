"""
FeatureRegionGrow3D Interactive Segmenter.

Local seeded 3D region growing in *feature space* (intensity, smoothed
intensity, gradient magnitude, Laplacian, optional Frangi vesselness).
Designed as an annotation-assistance tool — not a deep-learning model —
that resists leakage between nearby structures with similar raw intensity
(e.g. two adjacent vessels or touching cells).

For each positive seed point the segmenter:
  1. Extracts a local cube of side ``window_size`` centred on the seed.
  2. Computes per-voxel feature images inside the cube (only inside the
     cube — never on the full volume) and stacks them into an
     ``(N_features, Z, Y, X)`` array ``F``.
  3. Records the seed feature vector ``F0 = F[:, seed]``.
  4. Region-grows from the seed using a BFS queue, accepting a neighbour
     voxel when ``||F[:, n] - F0|| < feature_threshold``.
  5. Pastes the resulting binary mask back into a full-sized output volume.

Results from multiple seeds are combined by union.  Shapes are ignored
(matching how SAM3D / RegionGrow3D treat them as optional context only).
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from scipy.ndimage import gaussian_filter, laplace, sobel

    _is_available = True
except ImportError:
    gaussian_filter = None
    laplace = None
    sobel = None
    _is_available = False
    print("Warning: scipy not installed. FeatureRegionGrow3D will not work.")

try:
    from skimage.filters import frangi

    _has_frangi = True
except ImportError:
    frangi = None
    _has_frangi = False


@dataclass
class FeatureRegionGrow3D(InteractiveSegmenterBase):
    """
    Local seeded 3D region-growing segmenter in feature space.

    Click one or more points; each becomes a seed for a BFS region grow
    restricted to a local cube around it, accepting neighbours whose
    feature-vector distance to the seed is below ``feature_threshold``.
    Returns a binary uint8 mask the same shape as the input volume.
    """

    # Opt-in: the host app may re-run segment() with the last seed points
    # whenever any harvested parameter changes, giving live visual feedback.
    supports_live_param_update = True

    instructions = """
Instructions for Feature Region Grow 3D:
1. Click one or more points inside the structure of interest.
2. A local cube of side `window_size` is taken around each seed.
3. Per-voxel features are computed inside the cube:
     - raw intensity I
     - Gaussian-smoothed intensity G (sigma = `gaussian_sigma`)
     - gradient magnitude (optional, `use_gradient`)
     - Laplacian (optional, `use_laplacian`)
     - Frangi vesselness (optional, `use_vesselness`)
4. The seed's feature vector F0 is recorded.
5. BFS grows the region, accepting neighbours with
   ||F - F0|| < feature_threshold, where each feature is z-score
   standardized inside the cube so the threshold is in σ-units
   (typical useful range: 0.5 – 3.0).
6. `connectivity` chooses 6 / 18 / 26 neighbours.
7. Multiple positive points are combined by union.
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

    feature_threshold: float = field(
        default=1.5,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "default": 1.5,
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
        default=1.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 10.0,
            "step": 0.1,
            "default": 1.0,
        },
    )

    use_gradient: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "default": True,
        },
    )

    use_laplacian: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "default": True,
        },
    )

    use_vesselness: bool = field(
        default=False,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "default": False,
        },
    )

    def __init__(self):
        """Initialize the FeatureRegionGrow3D segmenter."""
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
        """Check if required dependencies (scipy) are available."""
        return _is_available

    # ------------------------------------------------------------------
    # Connectivity helper
    # ------------------------------------------------------------------
    @staticmethod
    def _neighbour_offsets(connectivity: int):
        """Return list of (dz, dy, dx) neighbour offsets for 6/18/26 connectivity."""
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    manh = abs(dz) + abs(dy) + abs(dx)
                    if connectivity <= 6 and manh != 1:
                        continue
                    if connectivity <= 18 and manh > 2:
                        continue
                    offsets.append((dz, dy, dx))
        return offsets

    # ------------------------------------------------------------------
    # Feature computation (cube-local only)
    # ------------------------------------------------------------------
    def _compute_features(self, cube_f):
        """Compute per-voxel feature stack inside a single cube.

        Returns an array of shape (N_features, Z, Y, X), float32, with each
        channel **z-score standardized within the cube** (mean 0, std 1).
        Without this normalization the Euclidean distance is dominated by
        whichever feature happens to have the largest absolute scale
        (typically raw intensity for uint16 data) and ``feature_threshold``
        becomes meaningless.  After standardization the threshold has the
        same units as a per-feature standard deviation, so values around
        1–3 give intuitive results regardless of input bit depth.

        Always includes raw intensity and Gaussian-smoothed intensity;
        gradient / Laplacian / vesselness toggle via the bool fields.
        """
        feats = [cube_f]

        if float(self.gaussian_sigma) > 0:
            gsmoothed = gaussian_filter(
                cube_f, sigma=float(self.gaussian_sigma)
            )
        else:
            gsmoothed = cube_f
        feats.append(gsmoothed)

        if bool(self.use_gradient):
            gz = sobel(gsmoothed, axis=0)
            gy = sobel(gsmoothed, axis=1)
            gx = sobel(gsmoothed, axis=2)
            grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
            feats.append(grad_mag.astype(np.float32, copy=False))

        if bool(self.use_laplacian):
            lap = laplace(gsmoothed)
            feats.append(lap.astype(np.float32, copy=False))

        if bool(self.use_vesselness) and _has_frangi:
            try:
                ves = frangi(gsmoothed).astype(np.float32, copy=False)
                feats.append(ves)
            except (ValueError, RuntimeError) as e:
                print(
                    f"FeatureRegionGrow3D: vesselness failed ({e}); skipping."
                )

        stack = np.stack(feats, axis=0).astype(np.float32, copy=False)

        # Per-feature z-score within the cube so all channels contribute
        # comparably to the Euclidean distance.  eps guards constant cubes.
        means = stack.mean(axis=(1, 2, 3), keepdims=True)
        stds = stack.std(axis=(1, 2, 3), keepdims=True)
        eps = np.float32(1e-6)
        stack = (stack - means) / (stds + eps)
        return stack

    # ------------------------------------------------------------------
    # BFS region grow in feature space (cube-local)
    # ------------------------------------------------------------------
    def _grow_from_seed(self, features, seed_local, offsets, threshold):
        """Grow a binary mask from ``seed_local`` using BFS in feature space.

        Args:
            features: (N_features, Z, Y, X) float32 array.
            seed_local: (z, y, x) tuple inside the cube.
            offsets: list of neighbour offsets.
            threshold: float, accept neighbour if ||F - F0|| < threshold.

        Returns:
            Boolean (Z, Y, X) array marking accepted voxels.
        """
        _, d, h, w = features.shape
        accepted = np.zeros((d, h, w), dtype=bool)
        sz, sy, sx = seed_local
        if not (0 <= sz < d and 0 <= sy < h and 0 <= sx < w):
            return accepted

        f0 = features[:, sz, sy, sx].astype(np.float32, copy=False)
        accepted[sz, sy, sx] = True

        queue = deque()
        queue.append((sz, sy, sx))
        thr2 = float(threshold) * float(threshold)

        while queue:
            cz, cy, cx = queue.popleft()
            for dz, dy, dx in offsets:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if not (0 <= nz < d and 0 <= ny < h and 0 <= nx < w):
                    continue
                if accepted[nz, ny, nx]:
                    continue
                diff = features[:, nz, ny, nx] - f0
                # squared Euclidean keeps it sqrt-free
                if float(np.dot(diff, diff)) < thr2:
                    accepted[nz, ny, nx] = True
                    queue.append((nz, ny, nx))

        return accepted

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform seeded 3D feature-space region growing.

        Args:
            image (numpy.ndarray): 3D (ZYX) or 4D (ZYXC) input volume.
            points (list, optional): Seed points. Last 3 components are (Z, Y, X).
            shapes (list, optional): Ignored (kept for API compatibility).
            **kwargs: Unused.

        Returns:
            numpy.ndarray: uint8 binary mask, same spatial shape as input.
        """
        if not _is_available:
            print(
                "FeatureRegionGrow3D: dependencies unavailable, returning None"
            )
            return None

        if image.ndim not in (3, 4):
            raise ValueError(
                f"FeatureRegionGrow3D requires 3D image. Got shape: {image.shape}"
            )

        # Collapse multichannel (ZYXC) to grayscale (ZYX).
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
            print("FeatureRegionGrow3D: no seed points provided")
            self.last_roi_bbox = None
            return full_mask

        half = max(1, int(self.window_size) // 2)
        offsets = self._neighbour_offsets(int(self.connectivity))
        threshold = float(self.feature_threshold)

        # Track aggregate bbox of all ROIs touched (for last_roi_bbox).
        z_lo = y_lo = x_lo = None
        z_hi = y_hi = x_hi = None

        for raw_pt in points:
            pt = np.asarray(raw_pt)
            if pt.size < 3:
                # 2D point — skip (this segmenter needs a Z coordinate).
                continue
            cz = int(round(float(pt[-3])))
            cy = int(round(float(pt[-2])))
            cx = int(round(float(pt[-1])))

            if not (0 <= cz < d and 0 <= cy < h and 0 <= cx < w):
                print(
                    f"FeatureRegionGrow3D: seed ({cz},{cy},{cx}) outside volume — skipped"
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
            try:
                features = self._compute_features(cube_f)
            except (ValueError, RuntimeError) as e:
                print(f"FeatureRegionGrow3D: feature computation failed: {e}")
                continue

            seed_local = (cz - z0, cy - y0, cx - x0)
            try:
                grown = self._grow_from_seed(
                    features, seed_local, offsets, threshold
                )
            except (ValueError, IndexError) as e:
                print(f"FeatureRegionGrow3D: grow failed at {seed_local}: {e}")
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
                f"FeatureRegionGrow3D: seed=({cz},{cy},{cx}) "
                f"cube z=[{z0}:{z1}] y=[{y0}:{y1}] x=[{x0}:{x1}] "
                f"n_feats={features.shape[0]} "
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
        return InteractiveSegmenterBase.register_framework(
            "FeatureRegionGrow3D", cls
        )
