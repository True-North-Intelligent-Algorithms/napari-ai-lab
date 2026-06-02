"""
sphere_mask_fit.py — Smooth star-convex sphere fitting from a 3D binary mask.

Given a 3D binary mask whose per-plane regions have already been filtered for
circularity (e.g. by SAMSphere3D), this utility:

  1. Measures a circle (centroid + equivalent radius) on every non-empty plane.
  2. Locates the *equatorial* plane — the one with the largest radius.
     Its weighted-median centroid becomes the XY centre of the fitted model.
  3. Fits a smooth, unimodal radius profile r(z):
       - Unimodal: r is non-increasing going away from the equatorial plane
         in both directions (single peak, no oscillations).
       - Smooth: a downward parabola is fitted to r²(z) with iterative
         outlier rejection, producing a noise-robust ellipsoidal profile.
         Falls back to the raw monotonic profile when the fit is degenerate.
  4. Rasterises the fitted radii as filled disks to produce a clean,
     smooth 3D mask of the same shape as the input.

The result is not a perfect sphere but a star-convex, sphere-*like* shape
whose cross-sections follow a single-peaked profile.  Upper and lower halves
are fitted independently so the shape can be lopsided (anisotropic in Z).

Public API
----------
fit_sphere_to_mask(mask, profile_outlier_sigma=2.0, min_planes=3,
                   debug=False) -> np.ndarray
"""

import numpy as np

try:
    from skimage.measure import label as sk_label
    from skimage.measure import regionprops

    _skimage_ok = True
except ImportError:
    regionprops = None
    sk_label = None
    _skimage_ok = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _measure_planes(mask):
    """Return per-plane circle measurements from a binary 3D mask.

    For each non-empty Z-plane the largest connected component is measured.
    Radius is the *equivalent radius* derived from area: r = sqrt(area / π).

    Returns
    -------
    list of (z, cy, cx, r)  — one entry per non-empty plane, sorted by z.
    Returns [] when scikit-image is unavailable.
    """
    if not _skimage_ok:
        return []

    planes = []
    for z in range(mask.shape[0]):
        plane = mask[z] > 0
        if not plane.any():
            continue
        labeled = sk_label(plane)
        props = regionprops(labeled)
        if not props:
            continue
        largest = max(props, key=lambda p: p.area)
        cy, cx = largest.centroid
        r = float(np.sqrt(largest.area / np.pi))
        planes.append((z, float(cy), float(cx), r))

    return sorted(planes, key=lambda t: t[0])


def _weighted_median(values, weights):
    """Weighted median of 1-D array."""
    order = np.argsort(values)
    vals_sorted = np.asarray(values)[order]
    w_sorted = np.asarray(weights)[order]
    cumw = np.cumsum(w_sorted)
    cutoff = cumw[-1] * 0.5
    idx = np.searchsorted(cumw, cutoff)
    return float(vals_sorted[min(idx, len(vals_sorted) - 1)])


def _enforce_monotonic(zs, rs, z_peak_idx):
    """Return radii enforced to be non-increasing away from the peak.

    Uses a cumulative minimum: going upward from peak, r[i] = min(r[i], r[i-1]);
    going downward from peak, r[i] = min(r[i], r[i+1]).
    """
    rs = np.array(rs, dtype=np.float64)
    n = len(rs)

    # Downward direction (indices < z_peak_idx)
    for i in range(z_peak_idx - 1, -1, -1):
        rs[i] = min(rs[i], rs[i + 1])

    # Upward direction (indices > z_peak_idx)
    for i in range(z_peak_idx + 1, n):
        rs[i] = min(rs[i], rs[i - 1])

    return rs


def _fit_parabola_profile(zs, rs, profile_outlier_sigma=2.0):
    """Fit r²(z) = c0 + c1·z + c2·z² with iterative outlier rejection.

    Returns the fitted radii array (same length as zs), or None if the fit
    is degenerate (c2 ≥ 0 or negative discriminant).

    Uses unit weights (quality not available from mask measurements, unlike
    Hough).  Outliers are down-weighted by the MAD criterion.
    """
    if len(zs) < 3:
        return None

    z64 = np.asarray(zs, dtype=np.float64)
    target = np.asarray(rs, dtype=np.float64) ** 2
    weights = np.ones(len(zs), dtype=np.float64)

    c0 = c1 = c2 = None
    for _ in range(4):
        if (weights > 0).sum() < 3:
            return None
        X = np.stack([np.ones_like(z64), z64, z64 * z64], axis=1)
        W = np.diag(weights)
        try:
            coeffs, *_ = np.linalg.lstsq(W @ X, W @ target, rcond=None)
        except np.linalg.LinAlgError:
            return None
        c0_, c1_, c2_ = coeffs
        if c2_ >= 0:
            return None  # not a downward parabola

        preds = c0_ + c1_ * z64 + c2_ * z64 * z64
        res = target - preds
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-9
        sigma = 1.4826 * mad
        keep = np.abs(res) <= profile_outlier_sigma * sigma
        new_weights = keep.astype(np.float64)
        if np.array_equal(new_weights > 0, weights > 0):
            c0, c1, c2 = c0_, c1_, c2_
            break
        weights = new_weights

    if c0 is None or c2 is None or c2 >= 0:
        return None

    # Extract sphere parameters
    z0 = -c1 / (2.0 * c2)
    a2 = c0 - c2 * z0 * z0
    b2 = -a2 / c2
    if a2 <= 0 or b2 <= 0:
        return None

    a = float(np.sqrt(a2))
    b = float(np.sqrt(b2))

    fitted = []
    for z in z64:
        inside = 1.0 - ((z - z0) / b) ** 2
        fitted.append(max(0.0, a * float(np.sqrt(max(0.0, inside)))))

    return np.array(fitted)


def _rasterize_disk(canvas, cy, cx, r):
    """OR a filled disk centred at (cy, cx) with radius r into a 2D canvas."""
    h, w = canvas.shape
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(h, int(np.ceil(cy + r)) + 1)
    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(w, int(np.ceil(cx + r)) + 1)
    if y1 <= y0 or x1 <= x0:
        return
    ys = np.arange(y0, y1, dtype=np.float64)[:, None]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :]
    canvas[y0:y1, x0:x1] |= (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_sphere_to_mask(
    mask,
    profile_outlier_sigma: float = 2.0,
    min_planes: int = 3,
    debug: bool = False,
) -> np.ndarray:
    """Fit a smooth star-convex sphere-like shape to a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray (Z, Y, X)  — binary or uint8 input mask.
    profile_outlier_sigma : float
        Threshold (in MAD units) for outlier rejection during the parabolic
        profile fit.  Higher = more lenient, lower = stricter.
    min_planes : int
        Minimum number of non-empty Z-planes required to attempt fitting.
        If fewer planes are found the input mask is returned unchanged.
    debug : bool
        If True, print diagnostic information.

    Returns
    -------
    np.ndarray (Z, Y, X) uint8 — the fitted mask.  Returns the input mask
    unchanged when fitting is not possible (too few planes, degenerate fit,
    or scikit-image unavailable).
    """
    if mask.ndim != 3:
        return mask

    if not _skimage_ok:
        if debug:
            print(
                "fit_sphere_to_mask: scikit-image unavailable, returning input."
            )
        return mask

    planes = _measure_planes(mask)

    if len(planes) < min_planes:
        if debug:
            print(
                f"fit_sphere_to_mask: only {len(planes)} plane(s) detected "
                f"(min_planes={min_planes}); returning input mask."
            )
        return mask

    zs = np.array([p[0] for p in planes], dtype=np.float64)
    cys = np.array([p[1] for p in planes])
    cxs = np.array([p[2] for p in planes])
    rs = np.array([p[3] for p in planes])

    # --- Equatorial plane: largest radius ---------------------------------
    peak_idx = int(np.argmax(rs))
    z_peak = int(zs[peak_idx])

    # Robust XY centre: weighted median, weight by radius (larger circles
    # contribute more — they are usually more reliable).
    w = rs / (rs.sum() + 1e-9)
    cy_center = _weighted_median(cys, w)
    cx_center = _weighted_median(cxs, w)

    if debug:
        print(
            f"fit_sphere_to_mask: equator z={z_peak} r={rs[peak_idx]:.1f} "
            f"center=({cy_center:.1f}, {cx_center:.1f}) "
            f"planes={len(planes)}"
        )

    # --- Radius profile ---------------------------------------------------
    # 1. Try smooth parabolic fit.
    fitted_rs = _fit_parabola_profile(
        zs, rs, profile_outlier_sigma=profile_outlier_sigma
    )

    if fitted_rs is not None:
        if debug:
            print("fit_sphere_to_mask: using smooth parabolic profile.")
        final_rs = fitted_rs
    else:
        # 2. Fall back: enforce monotonic decrease from peak.
        if debug:
            print(
                "fit_sphere_to_mask: parabolic fit degenerate, "
                "using monotonic profile."
            )
        final_rs = _enforce_monotonic(zs, rs, peak_idx)

    # --- Rasterise --------------------------------------------------------
    d, h, w_img = mask.shape
    fitted_mask = np.zeros((d, h, w_img), dtype=np.uint8)

    for (z, _, _, _), r_fit in zip(planes, final_rs, strict=False):
        if r_fit < 0.5:
            continue
        _rasterize_disk(fitted_mask[int(z)], cy_center, cx_center, r_fit)

    if debug:
        kept = int((fitted_mask > 0).any(axis=(1, 2)).sum())
        print(f"fit_sphere_to_mask: rasterised {kept} plane(s).")

    return fitted_mask
