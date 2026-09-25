"""napari-menu entry points for ND AI Lab.

One entry per profile.  Each registers every segmenter and augmenter its
profile allows, then hands the profile to NDAILab so the GUI matches.

Why subclasses and not a factory function?  Napari's plugin loader only
performs viewer-injection for **class** widget contributions; plain functions
are assumed to be magicgui widgets and receive no viewer.  So a profile is
bound by declaring a class, and ``napari.yaml`` names it.

Adding a menu entry for a new profile is two steps: a subclass here, and a
command plus widget in ``napari.yaml``.

Scripts want ``launch_nd_ai_lab(..., profile=...)`` instead, which takes the
profile directly.
"""

from napari.viewer import Viewer

from .nd_ai_lab import NDAILab
from .register_all import register_all


class ProfiledAILab(NDAILab):
    """NDAILab bound to the profile named by the class attribute."""

    #: Profile name, or None to take NAPARI_AI_LAB_PROFILE, then "all".
    profile_name: str | None = None

    def __init__(self, viewer: Viewer):
        register_all(self.profile_name)
        super().__init__(viewer, profile=self.profile_name)


class NDAILabPlugin(ProfiledAILab):
    """Everything installed.

    Left unpinned so NAPARI_AI_LAB_PROFILE can still redirect this entry;
    without it the profile resolves to "all".
    """

    profile_name = None


class NDAILabInstance2D(ProfiledAILab):
    """2D instance segmentation, global segmenters restricted to scikit-ops."""

    profile_name = "2d-instance-skop"
