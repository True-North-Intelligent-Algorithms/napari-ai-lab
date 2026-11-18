"""tnia-ops' photo detectors: the people in a photo, and their race bibs.

The ones photo-enhance uses, so a model trained or tested here is judged on
the same boxes. They are tnia-ops ops, not scikit-ops ones, but run the same
way: through scikit-ops' runner, in tnia-ops' own "photo" environment, which
the runner builds as tnia-ops--photo the first time.

Each detector here stands alone. photo-enhance chains them -- bibs searched
for inside each person, at full size -- and that chaining is its logic, not
ai-lab's. detect_bibs here looks at the whole photo at once, which OWLv2 sees
at 960 pixels, so a small bib far back can be missed.
"""

from dataclasses import dataclass, field

from ...utilities.shared_models import shared_model
from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopDetector import SkopDetector, _float_field

# Optional: ai-lab without tnia-ops simply does not offer these.
try:
    from tnia_ops.detect import detect_bibs, detect_people
except ImportError:
    detect_bibs = detect_people = None


def _friendly(default, label, help_, lo, hi, step, kind):
    """A field the form shows under *label*, with *help_* as its tooltip."""
    return field(
        default=default,
        metadata={
            "type": kind,
            "param_type": "inference",
            "label": label,
            "help": help_,
            "min": lo,
            "max": hi,
            "step": step,
            "default": default,
        },
    )


#: The model People runs, fetched once into the shared models folder.
PEOPLE_MODEL = "yolo11m.pt"
PEOPLE_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
    + PEOPLE_MODEL
)


@dataclass
class TniaPeopleDetector(SkopDetector):
    """People: tnia-ops' detect_people, with nothing to set up.

    The friendly counterpart of the YOLO segmenter. Its model is fixed -- a
    COCO YOLO kept to its person class -- and lives in the per-user shared
    models folder, downloaded there the first time, so it works in any
    project with no model to choose. It only runs on colour photos, so there
    is no axis to choose either; the options are about people.
    """

    show_axis_choice = False

    conf: float = _friendly(
        0.35,
        "Confidence it's a person",
        "Lower finds more people, and more things that are not people.",
        0.0,
        1.0,
        0.05,
        "float",
    )
    keep_relative_height: float = _friendly(
        0.45,
        "Only the nearest people",
        "Drop people shorter than this fraction of the tallest one -- the "
        "ones further back. 0 keeps everyone.",
        0.0,
        1.0,
        0.05,
        "float",
    )
    min_height: int = _friendly(
        0,
        "Minimum height (pixels)",
        "Drop people shorter than this, in the photo's pixels. 0 keeps "
        "everyone.",
        0,
        10000,
        10,
        "int",
    )
    size: int = _friendly(
        1280,
        "Detail (photo size)",
        "The photo's long side is looked at this many pixels long. Higher "
        "finds people further back, more slowly.",
        320,
        4096,
        32,
        "int",
    )

    instructions = """
People:
• finds the people in a photo, with a box around each
• works in any project: its model downloads once, the first time, to
  ~/.napari_ai_lab/models
• hover over an option to see what it does
• runs in tnia-ops' photo environment, built on first use
    """

    def __post_init__(self):
        super().__post_init__()
        self._potential_axes = ["YXC"]
        self._supported_axes = ["YXC"]

    @property
    def op(self):
        return detect_people

    def get_recommended_axis(self) -> str:
        return "YXC"

    def _op_params(self) -> dict:
        return {
            "weights": str(shared_model(PEOPLE_MODEL, PEOPLE_MODEL_URL)),
            "size": self.size,
            "conf": self.conf,
            "keep_relative_height": self.keep_relative_height,
            "min_height": self.min_height,
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework("People", cls)


@dataclass
class TniaBibDetector(SkopDetector):
    """tnia-ops' detect_bibs: OWLv2, told in words to find race bibs."""

    prompt: str = field(
        default="a race bib number",
        metadata={
            "type": "str",
            "param_type": "inference",
            "choices": ["a race bib number", "a number on a shirt"],
            "default": "a race bib number",
        },
    )
    threshold: float = _float_field(0.15, step=0.01)
    overlap: float = _float_field(0.5)
    model: str = field(
        default="google/owlv2-base-patch16-ensemble",
        metadata={
            "type": "str",
            "param_type": "inference",
            "choices": [
                "google/owlv2-base-patch16-ensemble",
                "google/owlv2-large-patch14-ensemble",
            ],
            "default": "google/owlv2-base-patch16-ensemble",
        },
    )

    instructions = """
Bibs, OWLv2 (tnia-ops):
• zero-shot: finds what the prompt describes, no bib training needed
• also boxes some torsos and signs -- not as precise as a trained detector
• threshold - OWLv2's scores are low; 0.15 is a fair bib
• overlap - boxes overlapping a better one by more than this are dropped
• looks at the whole photo at once, so small bibs far back can be missed
• the model downloads on first use; runs in tnia-ops' photo environment
    """

    @property
    def op(self):
        return detect_bibs

    def get_recommended_axis(self) -> str:
        return "YXC"

    def _op_params(self) -> dict:
        return {
            "prompt": self.prompt,
            "threshold": self.threshold,
            "overlap": self.overlap,
            "model": self.model,
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "Bibs, OWLv2 (tnia-ops)", cls
        )
