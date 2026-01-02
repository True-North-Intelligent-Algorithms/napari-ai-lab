"""
Microsam + YOLO Global Segmenter

This module wraps an object-detection (YOLO) + MicroSAM pipeline into a
GlobalSegmenter that can be used in the framework. It detects candidate boxes
with a YOLO detector and then runs MicroSAM on those boxes to obtain instance
segmentations which are merged into a labeled mask.
"""

from dataclasses import dataclass, field

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try imports; availability checked via are_dependencies_available
try:
    from segment_everything.mask_detectors.microsam import microsam_detector
    from segment_everything.object_detectors.yolo_detector import YoloDetector
    from segment_everything.stacked_labels import StackedLabels
    from segment_everything.weights_helper import get_weights_path

    _is_seg_everything_available = True
except ImportError:
    _is_seg_everything_available = False


@dataclass
class MicroSamYoloSegmenter(GlobalSegmenterBase):
    """
    YOLO -> MicroSAM global segmenter.

    This segmenter first runs a YOLO-based object detector to generate boxes,
    then runs MicroSAM on those boxes and aggregates instance masks into a
    2D labeled image.
    """

    # Short instructions/help text
    instructions = """
Microsam + YOLO Automatic Segmentation:
• Detect candidate objects with YOLO
• Run MicroSAM on detected boxes to get instance masks
• Aggregate masks into a single labeled image
If not enough GPU memory, try setting one or both of `yolo_device` and
`microsam_device` to 'cpu'.
"""

    # Parameters
    weights_key: str = field(
        default="ObjectAwareModelHuggingFace",
        metadata={
            "type": "str",
            "choices": ["ObjectAwareModelHuggingFace", "Default"],
            "default": "ObjectAwareModelHuggingFace",
        },
    )

    yolo_model_name: str = field(
        default="ObjectAwareModelFromMobileSamV2",
        metadata={
            "type": "str",
            "choices": [
                "ObjectAwareModelFromMobileSamV2",
                "ObjectAwareTiny",
                "DefaultYolo",
            ],
            "default": "ObjectAwareModelFromMobileSamV2",
        },
    )

    yolo_device: str = field(
        default="cuda",
        metadata={
            "type": "str",
            "choices": ["cuda", "cpu"],
            "default": "cuda",
        },
    )

    microsam_model_type: str = field(
        default="vit_b_lm",
        metadata={
            "type": "str",
            "choices": ["vit_b_lm", "vit_l_lm", "vit_h_lm"],
            "default": "vit_b_lm",
        },
    )

    microsam_device: str = field(
        default="cuda",
        metadata={
            "type": "str",
            "choices": ["cuda", "cpu"],
            "default": "cuda",
        },
    )

    conf: float = field(
        default=0.2,
        metadata={
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.2,
        },
    )

    imgsz: int = field(
        default=1024,
        metadata={
            "type": "int",
            "min": 128,
            "max": 8192,
            "step": 64,
            "default": 1024,
        },
    )
    # Simple area filter parameters. These are integers and always have a
    # value (no confusing None). Defaults: min_area=0, max_area=uint16 max.
    # These are used to build a minimal stat_limits dict and call
    # `filter_3d_labels_multi` on the StackedLabels instance.
    min_area: int = field(
        default=0,
        metadata={
            "type": "int",
            "min": 0,
            "max": int(np.iinfo(np.uint16).max),
            "step": 1,
            "default": 0,
        },
    )

    max_area: int = field(
        default=int(np.iinfo(np.uint16).max),
        metadata={
            "type": "int",
            "min": 0,
            "max": int(np.iinfo(np.uint16).max),
            "step": 1,
            "default": int(np.iinfo(np.uint16).max),
        },
    )

    def __post_init__(self):
        super().__init__()

    @property
    def supported_axes(self):
        return ["YXC", "YX", "ZYX"]

    def are_dependencies_available(self):
        return _is_seg_everything_available

    def get_version(self):
        parts = []
        if _is_seg_everything_available:
            parts.append("segment_everything")
        return ",".join(parts) if parts else None

    def get_parameters_dict(self):
        """
        Return current parameter values as a dict (same format as other segmenters).
        """
        return {
            "weights_key": self.weights_key,
            "yolo_model_name": self.yolo_model_name,
            "yolo_device": self.yolo_device,
            "microsam_model_type": self.microsam_model_type,
            "microsam_device": self.microsam_device,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "min_area": self.min_area,
            "max_area": self.max_area,
        }

    def segment(self, image, **kwargs):
        """
        Segment the given image and return a labeled mask (uint16).
        """

        # Default to CUDA when a device isn't explicitly provided. If you hit
        # out-of-memory errors on the GPU, set one or both devices to 'cpu'.
        y_dev = self.yolo_device if self.yolo_device is not None else "cuda"

        ms_dev = (
            self.microsam_device
            if self.microsam_device is not None
            else "cuda"
        )

        # Instantiate YOLO detector
        weights_path = str(get_weights_path(self.weights_key))
        yolo = YoloDetector(weights_path, self.yolo_model_name, device=y_dev)

        # Get boxes from YOLO
        bboxes = yolo.get_microsam_bboxes(
            image, conf=self.conf, imgsz=self.imgsz
        )

        # Use MicroSAM to generate masks for boxes
        detector = microsam_detector(
            model_type=self.microsam_model_type, device=ms_dev
        )
        detector.set_image(image)
        mask_list = detector.segment_boxes(bboxes)

        # Stack and convert to labels
        stacked = StackedLabels(mask_list)

        # Build a minimal stat_limits dict from min_area/max_area and
        # filter 3D labels. These are always integers (defaults above).
        stat_limits = {
            "area": {"min": int(self.min_area), "max": int(self.max_area)}
        }
        # call the stacked labels filter function (assumes the method
        # `filter_3d_labels_multi` exists on StackedLabels)

        stacked.filter_labels_3d_multi(stat_limits)
        labels = stacked.make_2d_labels(type="max")

        return labels.astype(np.uint16)

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "MicrosamYoloSegmenter", cls
        )
