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
    YoloCropAugmenter,
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
    ``launch_nd_ai_lab(profile=...)`` both come through here.
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

    # Offered in the profile's order, so its first is the one selected.
    _order_as_listed(GlobalSegmenterBase.registry, prof.global_segmenters)

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
    for aug in (SimpleAugmenter, AlbumentationsAugmenter, YoloCropAugmenter):
        if prof.allows(aug, "augmenter"):
            aug.register()


def _order_as_listed(registry, listed):
    """Put *registry* (name -> class) in the order *listed* names the classes.

    A profile that lists its segmenters lists them in the order it wants them
    offered: the Segment tab selects the first. Ones it does not list -- or
    every one, when *listed* is None -- keep the order they registered in.
    """
    if not listed:
        return
    rank = {name.lower(): i for i, name in enumerate(listed)}
    items = sorted(
        registry.items(),
        key=lambda item: rank.get(item[1].__name__.lower(), len(rank)),
    )
    registry.clear()
    registry.update(items)


def _register_skop_segmenters(prof):
    """Register the scikit-ops segmenters, if scikit-ops is installed."""
    try:
        from ..Segmenters.GlobalSegmenters.Cellpose3SkopSegmenter import (
            Cellpose3SkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.Cellpose4SkopSegmenter import (
            Cellpose4SkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.MicroSam2SkopSegmenter import (
            MicroSam2SkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.MicroSamSkopSegmenter import (
            MicroSamSkopSegmenter,
        )
        from ..Segmenters.GlobalSegmenters.StardistSkopSegmenter import (
            StardistSkopSegmenter,
        )
    except ImportError as exc:
        print(f"Note: scikit-ops segmenters not registered: {exc}")
        return

    from ..Segmenters.GlobalSegmenters.SkopDetector import (
        FastSamSkopDetector,
        ObjectAwareYoloSkopDetector,
        YoloSkopDetector,
    )

    # Registration order is list order, and the first is the default -- unless
    # the profile lists an order of its own (_order_as_listed): Cellpose (4,
    # Cellpose-SAM) first, as the one most people reach for.
    for seg in (
        Cellpose4SkopSegmenter,
        Cellpose3SkopSegmenter,
        StardistSkopSegmenter,
        MicroSamSkopSegmenter,
        MicroSam2SkopSegmenter,
        YoloSkopDetector,
        FastSamSkopDetector,
        ObjectAwareYoloSkopDetector,
    ):
        if prof.allows(seg, "global"):
            seg.register()

    # tnia-ops' detectors, run by the same runner. Optional: without tnia-ops
    # installed here they are not offered.
    from ..Segmenters.GlobalSegmenters import TniaOpsDetector as tnia

    if tnia.detect_people is None:
        print(
            "Note: tnia-ops detectors not registered: tnia-ops not installed"
        )
        return
    for seg in (tnia.TniaPeopleDetector, tnia.TniaBibDetector):
        if prof.allows(seg, "global"):
            seg.register()
