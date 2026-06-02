"""
AnisotropicSphereFit3D Interactive Segmenter.

Interactive segmenter for approximately-spherical objects in *anisotropic*
3D microscopy volumes where X/Y boundaries are strong but Z boundaries are
weak or missing.  The user clicks inside an object; the segmenter fits a
sphere (in physical coordinates) and returns a binary mask of voxels
inside that sphere.

Strategy:
  1.  Extract a local cube around the seed (only ever process this cube).
  2.  Estimate object intensity (mean/std) from a tiny neighbourhood
      around the click.
  3.  In the click Z-slice, cast radial rays from the click point and
      detect the strongest gradient response along each ray to get
      candidate boundary points; robustly fit a 2D circle (median /
      trimmed-median style — a tiny "poor man's RANSAC").
  4.  Convert to physical coordinates using ``xy_spacing`` and
      ``z_spacing`` so the sphere stays spherical regardless of voxel
      anisotropy.
  5.  Use the sphere prior to predict the expected XY radius at every
      Z-slice, optionally refining ``zc`` by a small search that
      maximises the intensity term on the predicted sphere shells
      (because raw axial edges are unreliable).
  6.  Rasterise the final sphere back into voxel space and paste the
      result into a full-sized output volume.

Multiple positive seeds are combined by union, matching the rest of the
interactive-segmenter family.
"""

from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from scipy.ndimage import gaussian_filter, map_coordinates, sobel

    _is_available = True
except ImportError:
    gaussian_filter = None
    map_coordinates = None
    sobel = None
    _is_available = False
    print(
        "Warning: scipy not installed. AnisotropicSphereFit3D will not work."
    )


@dataclass
class AnisotropicSphereFit3D(InteractiveSegmenterBase):
    """
    Local seeded sphere fitter for anisotropic 3D volumes.

    Click inside a roughly spherical object; the segmenter robustly fits
    a sphere in physical coordinates (so anisotropic voxel spacing is
    handled correctly) and returns the binary mask of voxels inside it.
    """

    # Opt-in: the host app re-runs segment() with the last seed points
    # whenever any harvested parameter changes (live preview).
    supports_live_param_update = True

    instructions = """
Instructions for Anisotropic Sphere Fit 3D:
1. Click ONE point inside a roughly spherical object.
2. A local cube of side `window_size` is taken around the seed.
3. Object intensity is estimated from voxels near the click.
4. Radial rays in the click Z-slice locate the X/Y boundary; a robust
   median circle fit gives (xc, yc, r_xy).
5. Voxel coordinates are converted to physical units using
   `xy_spacing` and `z_spacing` so the sphere is fit in real space.
6. If `allow_z_center_shift` is on, `zc` is refined by maximising the
   intensity-similarity score across predicted sphere shells.
7. The final sphere is rasterised back into voxel space.
Tips:
- For weak axial edges, increase `gradient_sigma` and rely on the
  intensity term — that's why this segmenter exists.
- Set `xy_spacing` / `z_spacing` to match your microscope (same units).
    """

    window_size: int = field(
        default=96,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 8,
            "max": 512,
            "step": 2,
            "default": 96,
        },
    )

    max_radius: float = field(
        default=80.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 1.0,
            "max": 1024.0,
            "step": 1.0,
            "default": 80.0,
        },
    )

    gradient_sigma: float = field(
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

    xy_spacing: float = field(
        default=1.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 1e-4,
            "max": 1000.0,
            "step": 0.01,
            "default": 1.0,
        },
    )

    z_spacing: float = field(
        default=1.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 1e-4,
            "max": 1000.0,
            "step": 0.01,
            "default": 1.0,
        },
    )

    intensity_weight: float = field(
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

    allow_radius_refinement: bool = field(
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

    allow_z_center_shift: bool = field(
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

    debug: bool = field(
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

    def __init__(self):
        """Initialize the AnisotropicSphereFit3D segmenter."""
        super().__init__()
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX", "ZYXC"]
        self.selected_axis = self._supported_axes[0]
        # ROI of the most recent fit (tuple of slices), mirrors
        # RegionGrow3D.last_roi_bbox so the host app can do fast updates.
        self.last_roi_bbox = None
        # Bag of diagnostics from the most recent segmentation, keyed by
        # seed index. Populated when ``debug`` is true.
        self.last_diagnostics = []

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

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _grayscale(image):
        """Collapse ZYXC -> ZYX exactly like RegionGrow3D does."""
        if image.ndim == 4:
            if image.shape[-1] == 3:
                return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            return np.mean(image, axis=-1)
        return image

    @staticmethod
    def _local_intensity_stats(cube, zc, yc, xc, radius=1):
        """Mean / std of a small box around the seed inside the cube."""
        d, h, w = cube.shape
        z0 = max(0, zc - radius)
        z1 = min(d, zc + radius + 1)
        y0 = max(0, yc - radius)
        y1 = min(h, yc + radius + 1)
        x0 = max(0, xc - radius)
        x1 = min(w, xc + radius + 1)
        patch = cube[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
        mean = float(patch.mean())
        std = float(patch.std()) + 1e-6
        return mean, std

    def _find_xy_boundary_points(self, slice_2d, yc, xc, max_r, n_rays=72):
        """Cast radial rays from (yc, xc) in a 2D slice, return boundary samples.

        Boundary is detected as the radius where the radial intensity
        derivative is most negative (transition from bright object to
        dark background) AND has the largest |gradient| response.
        Returns array of (y, x, score) tuples (score is the gradient
        magnitude, used for robust fitting).
        """
        h, w = slice_2d.shape
        smoothed = gaussian_filter(
            slice_2d.astype(np.float32, copy=False),
            sigma=float(self.gradient_sigma),
        )
        gy = sobel(smoothed, axis=0)
        gx = sobel(smoothed, axis=1)
        gmag = np.sqrt(gy * gy + gx * gx)

        max_r = max(2.0, float(max_r))
        n_samples = int(np.ceil(max_r)) + 1
        radii = np.linspace(0.0, max_r, n_samples).astype(np.float32)

        angles = np.linspace(
            0.0, 2.0 * np.pi, n_rays, endpoint=False, dtype=np.float32
        )
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # Sample each ray.
        ys = yc + np.outer(radii, sin_a)  # (n_samples, n_rays)
        xs = xc + np.outer(radii, cos_a)

        # Out-of-frame samples -> ignore later by masking.
        in_frame = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)

        # Sample intensity & gradient magnitude with bilinear interp.
        coords = np.stack(
            [np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], axis=0
        )
        intens = map_coordinates(smoothed, coords, order=1, mode="nearest")
        gsamp = map_coordinates(gmag, coords, order=1, mode="nearest")

        # Radial derivative: intensity at r+1 - r-1.
        d_intens = np.zeros_like(intens)
        d_intens[1:-1, :] = intens[2:, :] - intens[:-2, :]
        d_intens[~in_frame] = 0.0
        gsamp[~in_frame] = 0.0

        # We want the strongest *negative* derivative (bright -> dark)
        # combined with high |gradient|.  Score = -d_intens * gsamp,
        # restricted to non-trivial radii.
        score = -d_intens * gsamp
        score[:2, :] = -np.inf  # too close to seed

        best_idx = np.argmax(score, axis=0)  # per ray
        valid = np.isfinite(score[best_idx, np.arange(n_rays)]) & (
            score[best_idx, np.arange(n_rays)] > 0.0
        )

        best_r = radii[best_idx]
        by = yc + best_r * sin_a
        bx = xc + best_r * cos_a
        bscore = gsamp[best_idx, np.arange(n_rays)]

        pts = np.stack([by[valid], bx[valid], bscore[valid]], axis=1)  # (M, 3)
        return pts

    @staticmethod
    def _robust_circle_fit(pts, yc_init, xc_init, max_r):
        """Return (yc, xc, r) using a trimmed median estimator.

        Weighted by the gradient score so strong edges dominate.  Falls
        back to the seed centre + median radius when too few inliers.
        """
        if pts.shape[0] < 6:
            if pts.shape[0] == 0:
                return float(yc_init), float(xc_init), float(max_r * 0.5)
            r = float(
                np.median(np.hypot(pts[:, 0] - yc_init, pts[:, 1] - xc_init))
            )
            return float(yc_init), float(xc_init), float(min(r, max_r))

        ys, xs, sc = pts[:, 0], pts[:, 1], pts[:, 2]
        # Trim bottom 30% of scores to drop weakest "boundaries".
        thresh = np.quantile(sc, 0.3)
        keep = sc >= thresh
        if keep.sum() < 6:
            keep = np.ones_like(sc, dtype=bool)
        ys, xs = ys[keep], xs[keep]

        # Weighted centroid (closer to true center than the seed click).
        w = sc[keep]
        yc = float(np.sum(ys * w) / np.sum(w))
        xc = float(np.sum(xs * w) / np.sum(w))

        r_samples = np.hypot(ys - yc, xs - xc)
        # Use trimmed mean of inner 60% to ignore extreme outliers.
        lo, hi = np.quantile(r_samples, [0.2, 0.8])
        inliers = r_samples[(r_samples >= lo) & (r_samples <= hi)]
        if inliers.size == 0:
            r = float(np.median(r_samples))
        else:
            r = float(np.mean(inliers))
        r = float(min(max(r, 1.0), max_r))
        return yc, xc, r

    def _refine_z_center(
        self,
        cube,
        zc_init,
        yc,
        xc,
        r_xy_vox,
        obj_mean,
        obj_std,
    ):
        """Search a small Z range that best matches the sphere prior.

        At each candidate ``zc``, the predicted XY radius at slice z is
            r_z = sqrt(max(0, R_phys^2 - (z_phys - zc_phys)^2)) / xy_spacing
        We score each candidate by the average intensity-similarity of
        voxels inside the predicted disk (rewarded when |I - obj_mean|
        is small relative to obj_std).
        """
        d, _, _ = cube.shape
        R_phys = r_xy_vox * float(self.xy_spacing)
        z_phys_axis = np.arange(d, dtype=np.float32) * float(self.z_spacing)

        # Search +/- R_phys around zc_init (in slices).
        search_half = int(np.ceil(R_phys / float(self.z_spacing)))
        z_candidates = range(
            max(0, zc_init - search_half),
            min(d, zc_init + search_half + 1),
        )

        best_score = -np.inf
        best_zc = zc_init
        for zc_cand in z_candidates:
            zc_phys = float(zc_cand) * float(self.z_spacing)
            slice_score_sum = 0.0
            n_voxels = 0
            for z in range(d):
                dz_phys = z_phys_axis[z] - zc_phys
                r2 = R_phys * R_phys - dz_phys * dz_phys
                if r2 <= 0:
                    continue
                r_slice = np.sqrt(r2) / float(self.xy_spacing)
                if r_slice < 1.0:
                    continue
                disk_score, n = self._disk_intensity_score(
                    cube[z], yc, xc, r_slice, obj_mean, obj_std
                )
                slice_score_sum += disk_score
                n_voxels += n
            if n_voxels == 0:
                continue
            score = slice_score_sum / float(n_voxels)
            if score > best_score:
                best_score = score
                best_zc = zc_cand
        return best_zc, float(best_score)

    @staticmethod
    def _disk_intensity_score(slice_2d, yc, xc, r, obj_mean, obj_std):
        """Sum the intensity-similarity score over voxels inside a 2D disk."""
        h, w = slice_2d.shape
        y0 = max(0, int(np.floor(yc - r)))
        y1 = min(h, int(np.ceil(yc + r)) + 1)
        x0 = max(0, int(np.floor(xc - r)))
        x1 = min(w, int(np.ceil(xc + r)) + 1)
        if y1 <= y0 or x1 <= x0:
            return 0.0, 0
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist2 = (yy - yc) ** 2 + (xx - xc) ** 2
        mask = dist2 <= r * r
        patch = slice_2d[y0:y1, x0:x1].astype(np.float32, copy=False)
        if not mask.any():
            return 0.0, 0
        diffs = (patch[mask] - obj_mean) / obj_std
        # Higher score = closer to object intensity, in [0, 1].
        score = float(np.sum(np.exp(-0.5 * diffs * diffs)))
        return score, int(mask.sum())

    @staticmethod
    def _rasterize_sphere(shape, zc, yc, xc, R_phys, z_spacing, xy_spacing):
        """Boolean mask of voxels inside a physical-space sphere."""
        d, h, w = shape
        z = (np.arange(d, dtype=np.float32) - zc) * z_spacing
        y = (np.arange(h, dtype=np.float32) - yc) * xy_spacing
        x = (np.arange(w, dtype=np.float32) - xc) * xy_spacing
        zz = z[:, None, None]
        yy = y[None, :, None]
        xx = x[None, None, :]
        return (zz * zz + yy * yy + xx * xx) <= (R_phys * R_phys)

    # ==================================================================
    # Main entry point
    # ==================================================================
    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Fit a sphere to the object under each seed click.

        Args:
            image (numpy.ndarray): 3D (ZYX) or 4D (ZYXC) input volume.
            points (list, optional): Seed points. Last 3 components are (Z, Y, X).
            shapes (list, optional): Ignored.
            **kwargs: Unused.

        Returns:
            numpy.ndarray: uint8 binary mask, same spatial shape as input.
        """
        if not _is_available:
            print(
                "AnisotropicSphereFit3D: dependencies unavailable, returning None"
            )
            return None

        if image.ndim not in (3, 4):
            raise ValueError(
                f"AnisotropicSphereFit3D requires 3D image. Got shape: {image.shape}"
            )

        image_gray = self._grayscale(image)
        d, h, w = image_gray.shape[:3]
        full_mask = np.zeros((d, h, w), dtype=np.uint8)
        self.last_diagnostics = []

        if points is None or len(points) == 0:
            print("AnisotropicSphereFit3D: no seed points provided")
            self.last_roi_bbox = None
            return full_mask

        half = max(2, int(self.window_size) // 2)
        max_r_vox = float(self.max_radius)

        z_lo = y_lo = x_lo = None
        z_hi = y_hi = x_hi = None

        for i, raw_pt in enumerate(points):
            pt = np.asarray(raw_pt)
            if pt.size < 3:
                continue
            cz = int(round(float(pt[-3])))
            cy = int(round(float(pt[-2])))
            cx = int(round(float(pt[-1])))
            if not (0 <= cz < d and 0 <= cy < h and 0 <= cx < w):
                print(
                    f"AnisotropicSphereFit3D: seed ({cz},{cy},{cx}) outside volume — skipped"
                )
                continue

            z0 = max(0, cz - half)
            z1 = min(d, cz + half)
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)
            cube = image_gray[z0:z1, y0:y1, x0:x1].astype(
                np.float32, copy=False
            )
            if cube.size == 0:
                continue

            # Local coords of the seed inside the cube.
            sz = cz - z0
            sy = cy - y0
            sx = cx - x0

            # ---- 1. Object intensity stats ---------------------------
            obj_mean, obj_std = self._local_intensity_stats(
                cube, sz, sy, sx, radius=1
            )

            # ---- 2. XY boundary in the click slice -------------------
            click_slice = cube[sz]
            pts = self._find_xy_boundary_points(click_slice, sy, sx, max_r_vox)
            yc, xc, r_xy = self._robust_circle_fit(pts, sy, sx, max_r_vox)

            if not bool(self.allow_radius_refinement):
                # Pin (yc, xc) to the click if user doesn't want refinement.
                yc, xc = float(sy), float(sx)

            # ---- 3. Z center refinement (sphere prior) ---------------
            zc = sz
            z_score = 0.0
            if bool(self.allow_z_center_shift):
                zc, z_score = self._refine_z_center(
                    cube, sz, yc, xc, r_xy, obj_mean, obj_std
                )

            # ---- 4. Rasterise the sphere -----------------------------
            R_phys = r_xy * float(self.xy_spacing)
            sphere = self._rasterize_sphere(
                cube.shape,
                zc,
                yc,
                xc,
                R_phys,
                float(self.z_spacing),
                float(self.xy_spacing),
            )

            # ---- 5. Paste into full mask -----------------------------
            sub = full_mask[z0:z1, y0:y1, x0:x1]
            np.logical_or(sub.astype(bool), sphere, out=sphere)
            full_mask[z0:z1, y0:y1, x0:x1] = sphere.astype(np.uint8)

            # Aggregate bbox for fast repaint in the host app.
            z_lo = z0 if z_lo is None else min(z_lo, z0)
            y_lo = y0 if y_lo is None else min(y_lo, y0)
            x_lo = x0 if x_lo is None else min(x_lo, x0)
            z_hi = z1 if z_hi is None else max(z_hi, z1)
            y_hi = y1 if y_hi is None else max(y_hi, y1)
            x_hi = x1 if x_hi is None else max(x_hi, x1)

            diag = {
                "seed_voxel": (cz, cy, cx),
                "cube_origin": (z0, y0, x0),
                "obj_mean": obj_mean,
                "obj_std": obj_std,
                "n_boundary_points": int(pts.shape[0]),
                "fit_center_local": (zc, yc, xc),
                "fit_radius_voxels_xy": r_xy,
                "fit_radius_physical": R_phys,
                "z_score": z_score,
                "n_voxels_in_sphere": int(sphere.sum()),
            }
            self.last_diagnostics.append(diag)
            if bool(self.debug):
                print(
                    f"AnisotropicSphereFit3D[{i}]: seed=({cz},{cy},{cx}) "
                    f"obj_mean={obj_mean:.2f} obj_std={obj_std:.2f} "
                    f"n_edge_pts={int(pts.shape[0])} "
                    f"center_local=({zc:.1f},{yc:.1f},{xc:.1f}) "
                    f"r_xy_vox={r_xy:.1f} R_phys={R_phys:.2f} "
                    f"z_score={z_score:.3f} voxels={int(sphere.sum())}"
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
            "AnisotropicSphereFit3D", cls
        )
