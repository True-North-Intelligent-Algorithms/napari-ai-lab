"""Named sets of what ND AI Lab offers.

A profile narrows the app: which segmenters and augmenters register, and
which parts of the GUI are worth showing.  It never adds anything -- what a
profile allows still has to be registered and still has to suit the project.
Visibility is the intersection of the three, so a profile and a 2D project
cannot disagree about whether to show a 3D control.

Names are class names, matched case-insensitively, because the name a
segmenter registers under is not always its class name -- the registry lists
``MicroSamYoloSegmenter`` as "MicrosamYoloSegmenter" and
``StardistSkopSegmenter`` as "StarDist2D (scikit-ops)".

``None`` for a field means "everything available".  A tuple is an allow-list:
new segmenters do not appear in a curated profile until someone adds them,
which is the behaviour a teaching setup wants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Environment variable consulted when no profile is passed explicitly.
ENV_VAR = "NAPARI_AI_LAB_PROFILE"

DEFAULT = "all"


@dataclass(frozen=True)
class Profile:
    """One named configuration.  See the module docstring."""

    name: str
    description: str
    global_segmenters: tuple[str, ...] | None = None
    interactive_segmenters: tuple[str, ...] | None = None
    augmenters: tuple[str, ...] | None = None
    #: False hides the 3D box layer and its button, whatever the project is.
    show_3d: bool = True
    #: False leaves the Shapes layer uncreated. Shape prompts go with it.
    show_shapes: bool = True
    #: Optional GUI groups to keep, by key -- see FEATURES. None keeps all.
    features: tuple[str, ...] | None = None

    def allows(self, cls, kind: str) -> bool:
        """Whether *cls* is in this profile's list for *kind*.

        *kind* is "global", "interactive" or "augmenter".  A missing class --
        None, because an optional dependency is absent -- is never allowed.
        """
        if cls is None:
            return False
        allowed = {
            "global": self.global_segmenters,
            "interactive": self.interactive_segmenters,
            "augmenter": self.augmenters,
        }[kind]
        if allowed is None:
            return True
        return cls.__name__.lower() in {n.lower() for n in allowed}

    def allows_feature(self, key: str) -> bool:
        """Whether the GUI group *key* belongs in this profile. See FEATURES."""
        return self.features is None or key in self.features


#: GUI groups a profile can drop, and what each covers. The label widget
#: owns the widgets; this is the vocabulary the two ends agree on.
FEATURES = {
    "interactive-layer": "Interactive Layer combo and Add Interactive-Label Layer",
    "label-preview": "Label preview combo and Show labels in 2nd Napari",
    "local-ml": "Local Machine Learning",
    "edit-masks": "Edit Masks",
    "copy-predictions": "Copy predictions to labels",
}


PROFILES: dict[str, Profile] = {
    "all": Profile(
        name="all",
        description="Everything that is installed.",
    ),
    "2d-instance-skop": Profile(
        name="2d-instance-skop",
        description=(
            "2D instance segmentation, with the global segmenters restricted "
            "to the scikit-ops ones that run in their own environments. The "
            "interactive segmenters are host-side by necessity: interactive "
            "SAM keeps the image embedding resident between clicks, so it "
            "cannot run out of process."
        ),
        global_segmenters=(
            "StardistSkopSegmenter",
            "Cellpose3SkopSegmenter",
            "Cellpose4SkopSegmenter",
        ),
        interactive_segmenters=(
            "Otsu2D",
            "SAM3D",
        ),
        augmenters=(
            "SimpleAugmenter",
            "AlbumentationsAugmenter",
        ),
        show_3d=False,
        show_shapes=False,
        # None of the optional groups. "Active box size" is not among them --
        # it stays, because the box size is what tells you whether a box is
        # big enough for the patch size you mean to train at.
        features=(),
    ),
}


def get_profile(profile: str | Profile | None = None) -> Profile:
    """Resolve *profile*, falling back to the environment, then to "all".

    Accepts a Profile so a caller can pass one that is not in PROFILES.
    """
    if isinstance(profile, Profile):
        return profile
    name = profile or os.environ.get(ENV_VAR) or DEFAULT
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"Unknown profile {name!r}. Known profiles: {known}"
        ) from None


def list_profiles() -> list[str]:
    """Profile names, for a caller that wants to show the choices."""
    return sorted(PROFILES)
