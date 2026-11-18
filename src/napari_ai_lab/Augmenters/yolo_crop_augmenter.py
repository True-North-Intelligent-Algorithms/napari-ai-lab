"""YOLO crops: random-zoom crops around a box layer's boxes, for training.

Not a per-slice augmenter like the others: it makes a whole training set at
once, from every photo checked in one box layer, and writes it as
``patches/yolo_<layer>/`` for the Train tab. The work is in
``utilities.box_crops``; this class is the settings the Augment tab shows.
The tab sees ``makes_box_crops`` and calls ``write`` instead of ``augment``.
"""

from dataclasses import dataclass, field, fields

import numpy as np

from .augmenter_base import AugmenterBase


def _int(default, lo, hi, step=1):
    return field(
        default=default,
        metadata={
            "type": "int",
            "param_type": "augmentation",
            "min": lo,
            "max": hi,
            "step": step,
            "default": default,
        },
    )


@dataclass
class YoloCropAugmenter(AugmenterBase):
    """Random-zoom crops of a box layer's photos, written as a YOLO training set."""

    name: str = field(default="YOLO crops", init=False, repr=False)

    instructions: str = field(
        default="""
YOLO crops:
• Cuts crops from the full-size photos around the boxes of one box layer,
  each zoomed so its object is a random height between min and max object
• Each photo is also kept whole, so objects appear at every size
• Only photos marked Complete in that layer are used (Boxes tab), and
  photos with label boxes, inside the label boxes only
• Number of Patches - crops per photo with boxes, plus the photo whole
• Validation photos are held out whole, spread through the list
• No flips: numbers would be mirrored
• Written to patches/yolo_<layer>/ -- pick it in the Train tab
• Patch Size (XY) - side of each crop; train at this size
• min/max object - object height range in the crops, in pixels.
  Bibs: about 16-160. Runners: much bigger, e.g. 100-900
    """,
        init=False,
        repr=False,
    )

    #: Set from the tab's "Patch Size (XY)" and "Number of Patches" rows,
    #: as for the other augmenters, so not in the form.
    crop_size: int = 1024
    crops_per_photo: int = 16
    min_object: int = _int(16, 4, 4096, 4)
    max_object: int = _int(160, 8, 4096, 8)
    #: 0 holds back a tenth of the photos.
    val_photos: int = _int(0, 0, 10_000)
    deform: float = field(
        default=0.04,
        metadata={
            "type": "float",
            "param_type": "augmentation",
            "min": 0.0,
            "max": 0.2,
            "step": 0.01,
            "default": 0.04,
        },
    )
    seed: int = _int(0, 0, 99999)

    #: The Augment tab writes a crop set with ``write`` for this augmenter,
    #: from a box layer, instead of patches slice by slice.
    makes_box_crops = True

    def __post_init__(self):
        super().__init__(seed=self.seed)
        self._potential_axes = ["YXC", "YX"]
        self.supported_axes = ["YXC", "YX"]
        self.selected_axis = "YXC"

    @classmethod
    def register(cls):
        return AugmenterBase.register_framework("YOLO crops", cls)

    def get_parameters_dict(self):
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("param_type") == "augmentation"
        }

    def write(
        self, photos, out_dir, class_names=None, source="", progress=None
    ):
        """Write the crop set; see utilities.box_crops.write_box_crops."""
        from ..utilities.box_crops import write_box_crops

        if self.min_object >= self.max_object:
            raise ValueError("min object must be smaller than max object")
        return write_box_crops(
            photos,
            out_dir,
            class_names=class_names,
            crop_size=self.crop_size,
            per_photo=self.crops_per_photo,
            val_photos=self.val_photos,
            min_object=self.min_object,
            max_object=self.max_object,
            deform=self.deform,
            seed=self.seed,
            source=source,
            progress=progress,
        )

    def augment(self, im: np.ndarray, mask: np.ndarray, patch_size, axis=None):
        raise ValueError(
            "YOLO crops makes a training set from a box layer; "
            "it does not augment a mask."
        )
