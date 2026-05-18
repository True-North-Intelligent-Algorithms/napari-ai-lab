"""
Otsu3D Interactive Segmenter.

This module provides a 3D Otsu thresholding segmenter that works with
3D volumetric data and 3D multichannel images.
"""

from dataclasses import dataclass, field

import numpy as np
from skimage import filters, morphology

from .InteractiveSegmenterBase import InteractiveSegmenterBase


@dataclass
class Otsu3D(InteractiveSegmenterBase):
    """
    3D Otsu thresholding segmenter.

    Mirrors :class:`Otsu2D` but operates on 3D volumes.

    Behaviour:
      - If a shape (bounding box) is provided, its Y/X extents define the
        lateral ROI.  The Z extent is centred on the shape's Z plane and
        spans ``axial_roi_size`` slices.
      - Else if a point is provided, a ``lateral_roi_size`` x ``lateral_roi_size``
        x ``axial_roi_size`` ROI is centred on the point.
      - Else the whole volume is thresholded.
    """

    instructions = """
Instructions for Otsu 3D Segmentation:
1. Click a point or draw a box to threshold only a local 3D ROI.
2. Lateral ROI Size controls the XY side length (pixels).
3. Axial ROI Size controls the Z extent (slices) — also used when a box is given.
4. Without a point or shape, Otsu thresholding is applied to the entire volume.
5. The returned mask is always full-sized; non-ROI voxels are zero.
6. Optional binary opening / closing post-processing (open then close).
    """

    lateral_roi_size: int = field(
        default=30,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 0,
            "max": 500,
            "step": 1,
            "default": 30,
        },
    )

    axial_roi_size: int = field(
        default=10,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 10,
        },
    )

    apply_opening: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    apply_closing: bool = field(
        default=False,
        metadata={"type": "bool", "param_type": "inference", "default": False},
    )

    element_size: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 1,
            "max": 50,
            "step": 1,
            "default": 1,
        },
    )

    def __init__(self):
        """Initialize the Otsu3D segmenter."""
        super().__init__()
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX"]
        # Optional bbox of the most recent segmentation (tuple of slices) or None
        # if the whole volume was processed.  Callers may use it to restrict
        # mask-application work to the touched region.
        self.last_roi_bbox = None

    def segment(self, image, points=None, shapes=None, **kwargs):
        """Perform 3D Otsu segmentation. See class docstring for behaviour."""
        if image.ndim not in (3, 4):
            raise ValueError(
                f"Otsu3D requires 3D image. Got shape: {image.shape}"
            )

        # Convert multichannel (ZYXC) to grayscale (ZYX)
        use_multichannel = kwargs.get("use_multichannel", True)
        if image.ndim == 4 and use_multichannel:
            if image.shape[-1] == 3:  # RGB
                image_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                image_gray = np.mean(image, axis=-1)
        else:
            image_gray = image

        d, h, w = image_gray.shape[:3]
        full_mask = np.zeros((d, h, w), dtype=np.uint8)
        half_z = self.axial_roi_size // 2

        # Determine ROI bounds (priority: shape > point > full volume)
        z0 = z1 = y0 = y1 = x0 = x1 = None

        if shapes is not None and len(shapes) > 0:
            # Use the last shape as a box. Last 2 cols are Y, X; column [-3]
            # (if present) carries the Z plane the box was drawn on.
            box = np.asarray(shapes[-1])
            ys = box[:, -2]
            xs = box[:, -1]
            y0 = max(0, int(np.floor(np.min(ys))))
            y1 = min(h, int(np.ceil(np.max(ys))))
            x0 = max(0, int(np.floor(np.min(xs))))
            x1 = min(w, int(np.ceil(np.max(xs))))

            if box.shape[1] >= 3:
                zs = box[:, -3]
                z_min = float(np.min(zs))
                z_max = float(np.max(zs))
                if z_max - z_min > 0.5:
                    # True 3D box: use its Z extent directly.
                    z0 = max(0, int(np.floor(z_min)))
                    z1 = min(d, int(np.ceil(z_max)))
                else:
                    # 2D box drawn on one Z plane: extend by half_z.
                    cz = int(round(float(np.mean(zs))))
                    z0 = max(0, cz - half_z)
                    z1 = min(d, cz + half_z)
            else:
                cz = d // 2
                z0 = max(0, cz - half_z)
                z1 = min(d, cz + half_z)
            print(
                f"Otsu3D: using box ROI z=[{z0}:{z1}] y=[{y0}:{y1}] x=[{x0}:{x1}]"
            )
        elif points is not None and len(points) > 0:
            pt = np.asarray(points[-1])
            cz, cy, cx = int(pt[-3]), int(pt[-2]), int(pt[-1])
            half_xy = self.lateral_roi_size // 2

            z0 = max(0, cz - half_z)
            z1 = min(d, cz + half_z)
            y0 = max(0, cy - half_xy)
            y1 = min(h, cy + half_xy)
            x0 = max(0, cx - half_xy)
            x1 = min(w, cx + half_xy)

        if z0 is not None:
            roi = image_gray[z0:z1, y0:y1, x0:x1]

            if roi.size == 0:
                print("Otsu3D: ROI is empty, returning empty mask")
                self.last_roi_bbox = None
                return full_mask

            threshold = filters.threshold_otsu(roi)
            roi_mask = (roi > threshold).astype(np.uint8)
            roi_mask = self._apply_post_processing(roi_mask)
            full_mask[z0:z1, y0:y1, x0:x1] = roi_mask
            self.last_roi_bbox = (
                slice(z0, z1),
                slice(y0, y1),
                slice(x0, x1),
            )

            print(
                f"Otsu3D: ROI z=[{z0}:{z1}] y=[{y0}:{y1}] x=[{x0}:{x1}] "
                f"threshold={threshold:.2f}"
            )
        else:
            threshold = filters.threshold_otsu(image_gray)
            full_mask = (image_gray > threshold).astype(np.uint8)
            full_mask = self._apply_post_processing(full_mask)
            self.last_roi_bbox = None
            print(f"Otsu3D: Full-volume threshold {threshold:.2f}")

        return full_mask

    def _apply_post_processing(self, mask: np.ndarray) -> np.ndarray:
        """Apply binary open and/or close morphology to mask (open first)."""
        if not self.apply_opening and not self.apply_closing:
            return mask
        footprint = morphology.ball(self.element_size)
        result = mask.astype(bool)
        if self.apply_opening:
            result = morphology.binary_opening(result, footprint)
        if self.apply_closing:
            result = morphology.binary_closing(result, footprint)
        return result.astype(np.uint8)

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("Otsu3D", cls)
