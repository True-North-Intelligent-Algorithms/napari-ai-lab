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

from .GlobalSegmenterBase import GlobalSegmenterBase
from .SkopDetector import (
    PtModelPicker,
    SkopDetector,
    _float_field,
    _int_field,
)

# Optional: ai-lab without tnia-ops simply does not offer these.
try:
    from tnia_ops.detect import detect_bibs, detect_people
except ImportError:
    detect_bibs = detect_people = None


@dataclass
class TniaPeopleDetector(PtModelPicker, SkopDetector):
    """tnia-ops' detect_people: a COCO YOLO checkpoint kept to people."""

    model_file_path: str = field(
        default="",
        metadata={
            "type": "file",
            "param_type": "inference",
            "file_type": "file",
            "default": "",
        },
    )
    size: int = _int_field(1280, 320, 4096, 32)
    conf: float = _float_field(0.35)
    keep_relative_height: float = _float_field(0.45)

    instructions = """
People, YOLO (tnia-ops):
• model - a COCO YOLO .pt, e.g. yolo11m.pt, from the project's models/
  folder or browsed to with model file path
• size - the photo's long side is run at this; 1280 keeps people far back
• conf - lower finds more
• keep relative height - drop people shorter than this fraction of the
  tallest, the ones further back; 0 keeps everyone
• runs in tnia-ops' photo environment, built on first use
    """

    @property
    def op(self):
        return detect_people

    def _op_params(self) -> dict:
        return {
            "weights": self._weights(),
            "size": self.size,
            "conf": self.conf,
            "keep_relative_height": self.keep_relative_height,
        }

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "People, YOLO (tnia-ops)", cls
        )


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
