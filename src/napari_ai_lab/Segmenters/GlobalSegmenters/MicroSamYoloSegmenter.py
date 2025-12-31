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
    instructions: str = field(
        default="""
Microsam + YOLO Automatic Segmentation:
• Detect candidate objects with YOLO
• Run MicroSAM on detected boxes to get instance masks
• Aggregate masks into a single labeled image
If not enough GPU memory, try setting one or both of `yolo_device` and
`microsam_device` to 'cpu'.
""",
        metadata={"type": "str"},
    )

    # Parameters
    weights_key: str = field(default="ObjectAwareModelHuggingFace")
    yolo_model_name: str = field(default="ObjectAwareModelFromMobileSamV2")
    yolo_device: str | None = field(default=None)
    microsam_model_type: str = field(default="vit_b_lm")
    microsam_device: str | None = field(default=None)
    conf: float = field(default=0.2)
    imgsz: int = field(default=1024)

    def __post_init__(self):
        super().__init__()

    @property
    def supported_axes(self):
        return ["YX", "ZYX"]

    def are_dependencies_available(self):
        return _is_seg_everything_available

    def get_version(self):
        parts = []
        if _is_seg_everything_available:
            parts.append("segment_everything")
        return ",".join(parts) if parts else None

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
        labels = stacked.make_2d_labels(type="max")

        return labels.astype(np.uint16)

    @classmethod
    def register(cls):
        return GlobalSegmenterBase.register_framework(
            "MicrosamYoloSegmenter", cls
        )
