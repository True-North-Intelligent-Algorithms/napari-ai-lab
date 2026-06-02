"""
SAMSphere3D Interactive Segmenter.

Extends SAM3D with per-plane sphere quality checks.  The full SAM 3D
mask is computed first (including SAM's own volume propagation), then
each Z-plane is checked:

  1. Circularity — 4π·area / perimeter²  (1.0 = perfect circle).
     Planes below ``min_circularity`` are zeroed out.
  2. Centre proximity — Euclidean distance (pixels) between the plane
     centroid and the reference centroid established on the click plane.
     Planes beyond ``max_center_drift_xy`` are zeroed out.

All SAM initialisation, embeddings, and volume propagation logic is
inherited from SAM3D unchanged (DRY).
"""

from dataclasses import dataclass, field

import numpy as np

from napari_ai_lab.utilities.sphere_mask_fit import fit_sphere_to_mask

from .SAM3D import SAM3D

try:
    from skimage.measure import label as sk_label
    from skimage.measure import regionprops

    _is_skimage_available = True
except ImportError:
    regionprops = None
    sk_label = None
    _is_skimage_available = False
    print(
        "Warning: scikit-image not installed. "
        "SAMSphere3D sphere checks will be skipped."
    )


@dataclass
class SAMSphere3D(SAM3D):
    """
    SAM3D variant that post-filters the volumetric SAM mask plane-by-plane,
    keeping only Z-slices whose segmented region looks like a circle centred
    near the click-plane circle.

    All SAM infrastructure (predictor, embeddings, volume propagation) is
    inherited from SAM3D.  Only the ``segment()`` method is extended to add
    the per-plane sphere checks.
    """

    supports_live_param_update = True

    instructions = """
Instructions for SAMSphere3D:
1. Activate Points layer.
2. Click ONE point inside a roughly spherical object.
3. SAM segments the full 3D volume (same as SAM3D).
4. Each Z-plane is then checked:
   - Circularity (4π·area / perimeter²) must be ≥ min_circularity.
   - Centroid must be within max_center_drift_xy pixels of the
     reference centroid found on the click plane.
   Planes failing either check are zeroed out of the mask.
5. Press 'C' to commit, 'X' to erase, 'V' to toggle point polarity.
Tips:
- Lower min_circularity (e.g. 0.3) to accept less-round planes.
- Increase max_center_drift_xy to tolerate off-centre objects.
    """

    min_circularity: float = field(
        default=0.5,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "default": 0.5,
        },
    )

    max_center_drift_xy: float = field(
        default=20.0,
        metadata={
            "type": "float",
            "param_type": "inference",
            "harvest": True,
            "advanced": False,
            "training": False,
            "min": 0.0,
            "max": 500.0,
            "step": 1.0,
            "default": 20.0,
        },
    )

    fit_sphere: bool = field(
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
            "advanced": True,
            "training": False,
            "min": 0.5,
            "max": 10.0,
            "step": 0.5,
            "default": 2.0,
        },
    )

    def __init__(self):
        """Initialize SAMSphere3D — delegates fully to SAM3D.__init__."""
        super().__init__()

    # ------------------------------------------------------------------
    # Per-plane helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_circularity_and_centroid(binary_slice):
        """Compute circularity and centroid of the largest component in a
        binary 2D slice.

        Circularity = 4π·area / perimeter²  (1.0 for a perfect circle).

        Parameters
        ----------
        binary_slice : np.ndarray (H, W) bool

        Returns
        -------
        (circularity, cy, cx)
            circularity ∈ [0, 1]; (cy, cx) centroid of the largest region.
            Returns (0.0, None, None) when the slice is empty or degenerate.
        """
        if not _is_skimage_available:
            # Cannot check — treat every plane as passing.
            return 1.0, None, None

        if not binary_slice.any():
            return 0.0, None, None

        labeled = sk_label(binary_slice)
        props = regionprops(labeled)
        if not props:
            return 0.0, None, None

        # Use the largest connected component so a small stray blob doesn't
        # misrepresent the circularity.
        largest = max(props, key=lambda p: p.area)
        area = largest.area
        perim = largest.perimeter

        if perim < 1e-6:
            return 0.0, None, None

        circularity = min(1.0, (4.0 * np.pi * area) / (perim**2))
        cy, cx = largest.centroid
        return circularity, float(cy), float(cx)

    def _filter_mask_by_sphere_checks(self, mask, z_pos):
        """Zero out Z-planes that fail circularity or centre-drift checks.

        The reference circle is taken from the click plane (``z_pos``).
        Each other non-empty plane is tested:
          - circularity ≥ ``min_circularity``
          - centroid distance to reference centroid ≤ ``max_center_drift_xy``

        Parameters
        ----------
        mask : np.ndarray (Z, Y, X) uint8
        z_pos : int   Z-index of the click (reference plane)

        Returns
        -------
        np.ndarray — filtered mask, same shape and dtype as input.
        """
        if mask.ndim != 3:
            return mask  # 2D path — nothing to filter

        min_circ = float(self.min_circularity)
        max_drift = float(self.max_center_drift_xy)

        # Establish reference from click plane.
        ref_circ, ref_cy, ref_cx = self._slice_circularity_and_centroid(
            mask[z_pos] > 0
        )

        if ref_cy is None:
            print(
                f"SAMSphere3D: click plane z={z_pos} has no foreground; "
                "sphere checks skipped."
            )
            return mask

        if ref_circ < min_circ:
            print(
                f"SAMSphere3D: click plane z={z_pos} circularity={ref_circ:.3f} "
                f"below min_circularity={min_circ:.3f}; "
                "checks still applied to other planes."
            )

        filtered = mask.copy()
        n_kept = 0
        n_removed = 0

        for z in range(mask.shape[0]):
            plane_bin = mask[z] > 0
            if not plane_bin.any():
                continue  # already empty — nothing to filter

            circ, cy, cx = self._slice_circularity_and_centroid(plane_bin)

            if cy is None:
                filtered[z] = 0
                n_removed += 1
                continue

            drift = float(np.sqrt((cy - ref_cy) ** 2 + (cx - ref_cx) ** 2))
            fail_circ = circ < min_circ
            fail_drift = drift > max_drift

            if fail_circ or fail_drift:
                filtered[z] = 0
                n_removed += 1
                print(
                    f"SAMSphere3D: z={z} removed "
                    f"(circ={circ:.3f} {'FAIL' if fail_circ else 'ok'}, "
                    f"drift={drift:.1f}px {'FAIL' if fail_drift else 'ok'})"
                )
            else:
                n_kept += 1

        print(
            f"SAMSphere3D: ref z={z_pos} "
            f"centroid=({ref_cy:.1f}, {ref_cx:.1f}) circ={ref_circ:.3f} | "
            f"kept={n_kept} removed={n_removed}"
        )
        return filtered

    # ------------------------------------------------------------------
    # segment() — call SAM3D then apply sphere filter
    # ------------------------------------------------------------------

    def segment(self, image, points=None, shapes=None, **kwargs):
        """Run SAM3D segmentation then filter the result plane-by-plane.

        Delegates entirely to SAM3D.segment() for the SAM part, then
        post-processes with per-plane sphere quality checks.
        """
        mask = super().segment(image, points=points, shapes=shapes, **kwargs)

        if mask is None or mask.ndim != 3:
            # 2D result or failure — return as-is.
            return mask

        if points is None or len(points) == 0:
            return mask

        if len(points[0]) == 3:
            z_pos = int(points[0][0])
        else:
            # 2D click — no per-plane sphere check possible.
            return mask

        filtered = self._filter_mask_by_sphere_checks(mask, z_pos)

        if bool(self.fit_sphere):
            filtered = fit_sphere_to_mask(
                filtered,
                profile_outlier_sigma=float(self.profile_outlier_sigma),
                debug=True,
            )

        return filtered

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        from .InteractiveSegmenterBase import InteractiveSegmenterBase

        return InteractiveSegmenterBase.register_framework("SAMSphere3D", cls)
