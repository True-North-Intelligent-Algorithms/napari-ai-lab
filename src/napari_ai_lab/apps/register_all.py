"""
``register_all()`` — register the default set of augmenters and segmenters.

Use this when you want the standard set without listing each one (e.g. the
napari-menu entry point).  Cellpose and StarDist are registered as their
scikit-ops versions, which run in scikit-ops' own environments; the in-host
CellposeSegmenter and StardistSegmenter, and the Square2D toy, are left out.

Scripts that want selective registration (like ``launch_nd_ai_lab.py``)
should not call this function — they can import and register only what
they need.
"""

from ..Augmenters import (
    AlbumentationsAugmenter,
    SimpleAugmenter,
)
from ..Segmenters.GlobalSegmenters import (
    CellCastStardistSegmenter,
    MicroSamSegmenter,
    MicroSamYoloSegmenter,
    MonaiUNetSegmenter,
    MonaiUNetSegmenter3D,
    SkImageWatershedSegmenter,
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
)

# Segmenters left out of the default set for now.  Comment a line out to
# bring one back; nothing else needs changing.  This is the only switch, and
# it covers both entry points, because the napari menu and
# ``launch_nd_ai_lab(register_all=True)`` both come through here.
#
# Matched against the class name, case-insensitively, so the spelling the UI
# shows works too (it lists MicroSamYoloSegmenter as "MicrosamYoloSegmenter").
# A flat deny-list is enough while the set is small and static.  When it needs
# to depend on what is installed, or on the project, this is the thing to
# replace.
HIDDEN = {
    # global
    "MicroSamSegmenter",
    "MonaiUNetSegmenter",
    "MicroSamYoloSegmenter",
    "SkImageWatershedSegmenter",
    # interactive -- leaving Otsu2D and SAM3D only
    "Otsu3D",
    "SAMSphere3D",
    "RegionGrow3D",
    "FeatureRegionGrow3D",
    "AnisotropicSphereFit3D",
    "HoughSphereFit3D",
}


def _shown(seg):
    """Whether *seg* should be registered.  None means its deps are missing."""
    return seg is not None and seg.__name__.lower() not in {
        n.lower() for n in HIDDEN
    }


def register_all():
    """Register the default augmenters and segmenters."""
    _register_skop_segmenters()

    # Global segmenters — some are None when their optional deps are missing.
    for seg in (
        CellCastStardistSegmenter,
        ThresholdSegmenter,
        MicroSamSegmenter,
        MonaiUNetSegmenter,
        MonaiUNetSegmenter3D,
        MicroSamYoloSegmenter,
        SkImageWatershedSegmenter,
    ):
        if _shown(seg):
            seg.register()

    # Interactive segmenters
    for seg in (
        Otsu2D,
        Otsu3D,
        SAM3D,
        SAMSphere3D,
        RegionGrow3D,
        FeatureRegionGrow3D,
        AnisotropicSphereFit3D,
        HoughSphereFit3D,
    ):
        if _shown(seg):
            seg.register()

    # Augmenters
    SimpleAugmenter.register()
    AlbumentationsAugmenter.register()


def _register_skop_segmenters():
    """Register the scikit-ops segmenters, if scikit-ops is installed."""
    try:
        from ..Segmenters.GlobalSegmenters.Cellpose3SkopSegmenter import (
            Cellpose3SkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.Cellpose4SkopSegmenter import (
            Cellpose4SkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.StardistSkopSegmenter import (
            StardistSkopSegmenter,
        )
    except ImportError as exc:
        print(f"ℹ️  scikit-ops segmenters not registered: {exc}")
        return

    for seg in (
        StardistSkopSegmenter,
        Cellpose3SkopSegmenter,
        Cellpose4SkopSegmenter,
    ):
        if _shown(seg):
            seg.register()
