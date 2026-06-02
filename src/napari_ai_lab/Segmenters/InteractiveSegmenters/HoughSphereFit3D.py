"""
HoughSphereFit3D Interactive Segmenter.

Interactive segmenter for approximately-spherical objects in anisotropic
3D microscopy volumes.  Uses a 2D Hough circle transform on each Z-slice
near the click to fit a circle, then walks outward in Z until the fit
quality drops below a threshold — the resulting stack of circles forms
the sphere mask.

Designed for cases where X/Y boundaries are strong (Hough finds them
easily) but Z boundaries are weak (we stop when no plausible circle is
found rather than relying on axial edges).

Workflow per seed click (z, y, x):
  1. Extract a local cube of side ``window_size``.
  2. Run a 2D Hough circle transform on the click Z-slice within an
     ROI around (y, x), searching radii in [``min_radius``, ``max_radius``].
  3. Accept the best circle if its Hough accumulator score is at or
     above ``quality_threshold``.
  4. Walk +1, +2, ... and -1, -2, ... in Z up to ``max_planes`` each
     direction.  At each plane, search a narrower radius range around
     the previously accepted radius and require the centre to stay
     within ``max_center_drift`` of the click centre.  Stop in that
     direction the first time quality drops below threshold.
  5. Rasterise each accepted (zc, yc, xc, r) as a disk in voxel space
     and union them into the output volume.

Polarity:
  ``bright_on_dark``: edges go bright -> dark outward (default).
  ``dark_on_bright``: edges go dark -> bright outward (inverted image).
  ``auto``: pick whichever polarity gives a higher accumulator score
            in the click plane.
"""

from dataclasses import dataclass, field

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase

try:
    from scipy.ndimage import gaussian_filter
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    _is_available = True
except ImportError:
    gaussian_filter = None
    canny = None
    hough_circle = None
    hough_circle_peaks = None
    _is_available = False
    print(
        "Warning: scikit-image / scipy not installed. HoughSphereFit3D will not work."
    )


@dataclass
class HoughSphereFit3D(InteractiveSegmenterBase):
    """
    Per-plane Hough circle fitter for spherical objects.

    Click inside a roughly spherical object; the segmenter runs 2D Hough
    circle fits in the click plane and adjacent Z planes, growing the
    sphere outward until fit quality drops below the threshold.
    """

    supports_live_param_update = True

    instructions = """
Instructions for Hough Sphere Fit 3D:
1. Click ONE point inside a roughly spherical object.
2. A local cube of side `window_size` is taken around the seed.
3. The click Z-slice is edge-detected (Canny) and a 2D Hough circle
   transform finds the best circle in radii [min_radius, max_radius]
   near the click.
4. The segmenter walks outward in Z (up to `max_planes` each side),
   re-fitting on each plane with a narrowed radius range around the
   previous radius, and stops in each direction when quality drops
   below `quality_threshold`.
5. Each accepted disk is unioned into the output mask.
Tips:
- Increase `quality_threshold` (e.g. 0.4) for stricter fits.
- If the object is dark on a bright background, switch `polarity`.
- Set `polarity="auto"` to let the segmenter pick.
    """

    window_size: int = field(
        default=128,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 16,
            "max": 512,
            "step": 2,
            "default": 128,
        },
    )

    min_radius: int = field(
        default=10,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 2,
            "max": 500,
            "step": 1,
            "default": 10,
        },
    )

    max_radius: int = field(
        default=60,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 3,
            "max": 1000,
            "step": 1,
            "default": 60,
        },
    )

    max_planes: int = field(
        default=8,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0,
            "max": 200,
            "step": 1,
            "default": 8,
        },
    )

    quality_threshold: float = field(
        default=0.25,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.25,
        },
    )

    max_center_drift: int = field(
        default=6,
        metadata={
            "type": "int",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0,
            "max": 100,
            "step": 1,
            "default": 6,
        },
    )

    canny_sigma: float = field(
        default=2.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1,
            "default": 2.0,
        },
    )

    polarity: str = field(
        default="auto",
        metadata={
            "type": "str",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "choices": ["auto", "bright_on_dark", "dark_on_bright"],
            "default": "auto",
        },
    )

    enforce_smooth_profile: bool = field(
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

    profile_outlier_sigma: float = field(
        default=2.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.5,
            "max": 10.0,
            "step": 0.1,
            "default": 2.0,
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
        """Initialize the HoughSphereFit3D segmenter."""
        super().__init__()
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX", "ZYXC"]
        self.selected_axis = self._supported_axes[0]
        self.last_roi_bbox = None
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
        """Check if required dependencies (skimage, scipy) are available."""
        return _is_available

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _grayscale(image):
        if image.ndim == 4:
            if image.shape[-1] == 3:
                return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            return np.mean(image, axis=-1)
        return image

    @staticmethod
    def _normalize01(slice_2d):
        """Min-max normalize a 2D slice to [0, 1] float32 (robust to flat)."""
        s = slice_2d.astype(np.float32, copy=False)
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-12:
            return np.zeros_like(s)
        return (s - lo) / (hi - lo)

    def _edge_map(self, slice_2d, invert):
        """Canny edge map.  ``invert=True`` flips polarity (dark-on-bright)."""
        norm = self._normalize01(slice_2d)
        if invert:
            norm = 1.0 - norm
        return canny(norm, sigma=float(self.canny_sigma))

    def _hough_best_in_window(
        self,
        slice_2d,
        yc,
        xc,
        r_lo,
        r_hi,
        invert,
        center_window,
    ):
        """Run Hough circle on `slice_2d`, return best circle near (yc, xc).

        Returns (best_y, best_x, best_r, quality) where quality is in [0, 1].
        Quality is the peak accumulator value (skimage already normalises
        ``hough_circle`` accumulators per radius — peaks are roughly in
        [0, 1]).  Returns (None, None, None, 0.0) on failure.
        """
        h, w = slice_2d.shape
        r_lo = max(2, int(r_lo))
        r_hi = max(r_lo + 1, int(r_hi))
        if r_hi - r_lo < 1:
            return None, None, None, 0.0

        edges = self._edge_map(slice_2d, invert=invert)
        if not edges.any():
            return None, None, None, 0.0

        radii = np.arange(r_lo, r_hi + 1, dtype=int)
        try:
            accums = hough_circle(edges, radii)
        except (ValueError, RuntimeError) as e:
            print(f"HoughSphereFit3D: hough_circle failed: {e}")
            return None, None, None, 0.0

        # Restrict the centre search to a window around (yc, xc) so we
        # don't grab a different object that happens to be a stronger
        # circle.  We do this by zeroing the accumulator outside the
        # window before picking peaks.
        if center_window is not None and center_window > 0:
            yy = np.arange(h)[:, None]
            xx = np.arange(w)[None, :]
            window_mask = (np.abs(yy - yc) <= center_window) & (
                np.abs(xx - xc) <= center_window
            )
            if not window_mask.any():
                return None, None, None, 0.0
            accums = accums * window_mask[None, :, :]

        try:
            accum_peaks, cx_arr, cy_arr, r_arr = hough_circle_peaks(
                accums, radii, total_num_peaks=1, normalize=False
            )
        except (ValueError, RuntimeError) as e:
            print(f"HoughSphereFit3D: hough_circle_peaks failed: {e}")
            return None, None, None, 0.0
        if len(accum_peaks) == 0:
            return None, None, None, 0.0
        quality = float(accum_peaks[0])
        cy = float(cy_arr[0])
        cx = float(cx_arr[0])
        r = float(r_arr[0])
        return cy, cx, r, quality

    @staticmethod
    def _rasterize_disk(canvas, yc, xc, r):
        """OR a filled 2D disk centred at (yc, xc) with radius r into canvas."""
        h, w = canvas.shape
        y0 = max(0, int(np.floor(yc - r)))
        y1 = min(h, int(np.ceil(yc + r)) + 1)
        x0 = max(0, int(np.floor(xc - r)))
        x1 = min(w, int(np.ceil(xc + r)) + 1)
        if y1 <= y0 or x1 <= x0:
            return
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        mask = (ys - yc) ** 2 + (xs - xc) ** 2 <= r * r
        canvas[y0:y1, x0:x1] |= mask

    # ------------------------------------------------------------------
    # Unimodal smooth-profile fit
    # ------------------------------------------------------------------
    def _fit_smooth_profile(self, accepted):
        """Robustly fit r(z) = a * sqrt(1 - ((z - z0) / b)^2) and resample.

        This enforces:
          * a single peak in z (no oscillations or jagged protrusions),
          * smooth growth then shrinkage,
          * anisotropic (lopsided) z stretching via independent a, b,
          * outlier rejection (Hough mis-fits get dropped).

        Strategy:
          Squaring both sides gives r^2 = a^2 - (a^2 / b^2) * (z - z0)^2
          which is a parabola in z.  We fit the parabola
              r^2 = c0 + c1 * z + c2 * z^2
          with weighted least squares (weight = Hough quality), iterate
          while dropping points whose residual exceeds
          ``profile_outlier_sigma`` * MAD.  Then we extract
              z0 = -c1 / (2 c2),  a^2 = c0 - c2 * z0^2,  b^2 = -a^2 / c2.
          Centres (cy, cx) are smoothed by their weighted median so a
          single Hough mis-localisation doesn't shift the whole sphere.
        """
        if len(accepted) < 3:
            return accepted

        arr = np.array(
            [(z, cy, cx, r, q) for (z, cy, cx, r, q) in accepted],
            dtype=np.float32,
        )
        zs = arr[:, 0]
        cys = arr[:, 1]
        cxs = arr[:, 2]
        rs = arr[:, 3]
        qs = arr[:, 4]

        # Robust centre estimate: median weighted by quality.
        w = qs / max(qs.sum(), 1e-6)
        order_y = np.argsort(cys)
        cy_med = float(
            cys[order_y][np.searchsorted(np.cumsum(w[order_y]), 0.5)]
        )
        order_x = np.argsort(cxs)
        cx_med = float(
            cxs[order_x][np.searchsorted(np.cumsum(w[order_x]), 0.5)]
        )

        # Iterative weighted parabola fit on (z, r^2).
        target = rs.astype(np.float64) ** 2
        z64 = zs.astype(np.float64)
        weights = qs.astype(np.float64).copy()

        a2 = b2 = z0 = None
        for _ in range(4):
            if (weights > 0).sum() < 3:
                break
            X = np.stack([np.ones_like(z64), z64, z64 * z64], axis=1)
            W = np.diag(weights)
            try:
                coeffs, *_ = np.linalg.lstsq(W @ X, W @ target, rcond=None)
            except np.linalg.LinAlgError:
                break
            c0, c1, c2 = coeffs
            if c2 >= 0:
                # Not a downward parabola -> can't be a sphere profile.
                break
            z0 = -c1 / (2.0 * c2)
            a2 = c0 - c2 * z0 * z0
            b2 = -a2 / c2
            if a2 <= 0 or b2 <= 0:
                break

            # Residuals and robust outlier rejection (MAD scaled).
            preds = c0 + c1 * z64 + c2 * z64 * z64
            res = target - preds
            mad = np.median(np.abs(res - np.median(res))) + 1e-6
            sigma = 1.4826 * mad
            keep = np.abs(res) <= float(self.profile_outlier_sigma) * sigma
            new_weights = qs.astype(np.float64) * keep
            if np.array_equal(new_weights > 0, weights > 0):
                weights = new_weights
                break
            weights = new_weights

        if a2 is None or a2 <= 0 or b2 is None or b2 <= 0:
            # Couldn't fit a downward parabola — fall back to raw Hough.
            if self.debug:
                print(
                    "HoughSphereFit3D: smooth-profile fit failed, "
                    "using raw Hough results."
                )
            return accepted

        a = float(np.sqrt(a2))
        b = float(np.sqrt(b2))
        z0f = float(z0)

        # Resample r over the full z extent the original Hough run covered.
        z_min = int(np.floor(zs.min()))
        z_max = int(np.ceil(zs.max()))
        new_accepted = []
        for z in range(z_min, z_max + 1):
            inside = 1.0 - ((z - z0f) / b) ** 2
            if inside <= 0:
                continue
            r_fit = a * float(np.sqrt(inside))
            if r_fit < 1.0:
                continue
            new_accepted.append((int(z), cy_med, cx_med, r_fit, 1.0))

        if not new_accepted:
            return accepted

        if self.debug:
            print(
                f"HoughSphereFit3D: smooth profile a={a:.2f} b={b:.2f} "
                f"z0={z0f:.2f} planes={len(new_accepted)} "
                f"(raw {len(accepted)})"
            )
        return new_accepted

    # ==================================================================
    # Main entry point
    # ==================================================================
    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Fit a sphere (stack of Hough-fit circles) around each seed click.
        """
        if not _is_available:
            print("HoughSphereFit3D: dependencies unavailable, returning None")
            return None

        if image.ndim not in (3, 4):
            raise ValueError(
                f"HoughSphereFit3D requires 3D image. Got shape: {image.shape}"
            )

        image_gray = self._grayscale(image)
        d, h, w = image_gray.shape[:3]
        full_mask = np.zeros((d, h, w), dtype=np.uint8)
        self.last_diagnostics = []

        if points is None or len(points) == 0:
            print("HoughSphereFit3D: no seed points provided")
            self.last_roi_bbox = None
            return full_mask

        half = max(4, int(self.window_size) // 2)
        r_min = max(2, int(self.min_radius))
        r_max = max(r_min + 1, int(self.max_radius))
        max_planes = max(0, int(self.max_planes))
        q_thresh = float(self.quality_threshold)
        max_drift = max(0, int(self.max_center_drift))

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
                    f"HoughSphereFit3D: seed ({cz},{cy},{cx}) outside volume — skipped"
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

            sz = cz - z0
            sy = cy - y0
            sx = cx - x0

            # --- Polarity selection -----------------------------------
            polarity = self.polarity
            click_slice = cube[sz]
            if polarity == "auto":
                _, _, _, q_bod = self._hough_best_in_window(
                    click_slice,
                    sy,
                    sx,
                    r_min,
                    r_max,
                    invert=False,
                    center_window=max_drift,
                )
                _, _, _, q_dob = self._hough_best_in_window(
                    click_slice,
                    sy,
                    sx,
                    r_min,
                    r_max,
                    invert=True,
                    center_window=max_drift,
                )
                invert = q_dob > q_bod
                if self.debug:
                    print(
                        f"HoughSphereFit3D[{i}]: polarity auto "
                        f"bright_on_dark q={q_bod:.3f} "
                        f"dark_on_bright q={q_dob:.3f} "
                        f"-> {'dark_on_bright' if invert else 'bright_on_dark'}"
                    )
            else:
                invert = polarity == "dark_on_bright"

            # --- Click plane fit --------------------------------------
            cy0, cx0, r0, q0 = self._hough_best_in_window(
                click_slice,
                sy,
                sx,
                r_min,
                r_max,
                invert=invert,
                center_window=max_drift,
            )
            if cy0 is None or q0 < q_thresh:
                if self.debug:
                    print(
                        f"HoughSphereFit3D[{i}]: click-plane fit "
                        f"q={q0:.3f} below threshold {q_thresh}, skipping"
                    )
                continue

            accepted = [(sz, cy0, cx0, r0, q0)]

            # --- Walk +Z then -Z --------------------------------------
            for direction in (+1, -1):
                prev_y, prev_x, prev_r = cy0, cx0, r0
                for step in range(1, max_planes + 1):
                    z = sz + direction * step
                    if not (0 <= z < cube.shape[0]):
                        break
                    # Narrow radius range around prev_r; allow shrinkage
                    # as we move toward the pole of the sphere.
                    r_lo_plane = max(r_min, int(prev_r * 0.5))
                    r_hi_plane = min(r_max, int(prev_r * 1.05) + 1)
                    cy_p, cx_p, r_p, q_p = self._hough_best_in_window(
                        cube[z],
                        prev_y,
                        prev_x,
                        r_lo_plane,
                        r_hi_plane,
                        invert=invert,
                        center_window=max_drift,
                    )
                    if cy_p is None or q_p < q_thresh:
                        break
                    accepted.append((z, cy_p, cx_p, r_p, q_p))
                    prev_y, prev_x, prev_r = cy_p, cx_p, r_p

            # --- Smooth-profile fit (unimodal, anisotropic ellipsoid) ---
            if bool(self.enforce_smooth_profile) and len(accepted) >= 3:
                accepted = self._fit_smooth_profile(accepted)

            # --- Rasterise --------------------------------------------
            for z_local, yy, xx, rr, _q in accepted:
                self._rasterize_disk(
                    full_mask[z0 + z_local, y0:y1, x0:x1],
                    yy,
                    xx,
                    rr,
                )

            z_lo = z0 if z_lo is None else min(z_lo, z0)
            y_lo = y0 if y_lo is None else min(y_lo, y0)
            x_lo = x0 if x_lo is None else min(x_lo, x0)
            z_hi = z1 if z_hi is None else max(z_hi, z1)
            y_hi = y1 if y_hi is None else max(y_hi, y1)
            x_hi = x1 if x_hi is None else max(x_hi, x1)

            diag = {
                "seed_voxel": (cz, cy, cx),
                "cube_origin": (z0, y0, x0),
                "invert": invert,
                "n_planes_accepted": len(accepted),
                "click_quality": q0,
                "click_radius": r0,
                "click_center": (cy0, cx0),
                "planes": [(z + z0, q, r) for (z, _, _, r, q) in accepted],
            }
            self.last_diagnostics.append(diag)
            if self.debug:
                print(
                    f"HoughSphereFit3D[{i}]: seed=({cz},{cy},{cx}) "
                    f"invert={invert} click q={q0:.3f} r={r0:.1f} "
                    f"planes={len(accepted)} "
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
            "HoughSphereFit3D", cls
        )
