"""
``register_all()`` — try-register every available augmenter and segmenter.

Use this when you want the full set without listing each one (e.g. the
napari-menu entry point).  Scripts that want selective registration
(like ``launch_nd_ai_lab.py``) should not call this function — they can
import and register only what they need.
"""

from ..Augmenters import (
    AlbumentationsAugmenter,
    SimpleAugmenter,
)
from ..Segmenters.GlobalSegmenters import (
    CellCastStardistSegmenter,
    CellposeSegmenter,
    MicroSamSegmenter,
    MicroSamYoloSegmenter,
    MonaiUNetSegmenter,
    MonaiUNetSegmenter3D,
    StardistSegmenter,
    ThresholdSegmenter,
)
from ..Segmenters.InteractiveSegmenters import (
    SAM3D,
    AnisotropicSphereFit3D,
    FeatureRegionGrow3D,
    HoughSphereFit3D,
    Otsu2D,
    Otsu3D,
    RegionGrow3D,
    SAMSphere3D,
    Square2D,
)


def register_all():
    """Register every available augmenter and segmenter."""
    # Global segmenters — some are None when their optional deps are missing.
    for seg in (
        CellposeSegmenter,
        StardistSegmenter,
        CellCastStardistSegmenter,
        ThresholdSegmenter,
        MicroSamSegmenter,
        MonaiUNetSegmenter,
        MonaiUNetSegmenter3D,
        MicroSamYoloSegmenter,
    ):
        if seg is not None:
            seg.register()

    # Interactive segmenters
    for seg in (
        Square2D,
        Otsu2D,
        Otsu3D,
        SAM3D,
        SAMSphere3D,
        RegionGrow3D,
        FeatureRegionGrow3D,
        AnisotropicSphereFit3D,
        HoughSphereFit3D,
    ):
        seg.register()

    # Augmenters
    SimpleAugmenter.register()
    AlbumentationsAugmenter.register()
