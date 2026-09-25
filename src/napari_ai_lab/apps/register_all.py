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
    GlobalSegmenterBase,
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
    InteractiveSegmenterBase,
    Otsu2D,
    Otsu3D,
    RegionGrow3D,
    SAMSphere3D,
)
from .profiles import get_profile


def register_all(profile=None):
    """Register the augmenters and segmenters the profile allows.

    *profile* is a name, a Profile, or None -- see apps/profiles.py, which
    also documents the environment variable consulted when it is None. This
    is the only switch, and it covers both entry points: the napari menu and
    ``launch_nd_ai_lab(register_all=True)`` both come through here.
    """
    prof = get_profile(profile)
    print(f"Registering profile: {prof.name}")

    # Start from empty. The registries are class-level and live as long as the
    # process, so without this, opening one profile's widget and then another's
    # in the same napari session leaves the first profile's segmenters behind
    # -- a profile could only ever add. A script that registers extras itself
    # should do so after calling this.
    GlobalSegmenterBase.registry.clear()
    InteractiveSegmenterBase.registry.clear()

    _register_skop_segmenters(prof)

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
        if prof.allows(seg, "global"):
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
        if prof.allows(seg, "interactive"):
            seg.register()

    # Augmenters
    for aug in (SimpleAugmenter, AlbumentationsAugmenter):
        if prof.allows(aug, "augmenter"):
            aug.register()


def _register_skop_segmenters(prof):
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
        if prof.allows(seg, "global"):
            seg.register()
