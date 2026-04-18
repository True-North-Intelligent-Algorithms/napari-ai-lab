"""
Utilities module for napari-ai-lab.
"""

from .dl_util import (
    collect_training_data,
    compute_percentiles,
    divide_training_data,
    normalize_intensity,
    normalize_percentile,
)
from .image_util import (
    IMAGE_EXTENSIONS,
    collect_all_image_names,
    create_artifact_name,
    create_empty_instance_image,
    get_axis_info,
    get_axis_info_from_shape,
    get_current_slice_indices,
    get_ndim,
    get_supported_axes_from_shape,
    load_images_from_directory,
    pad_to_largest,
    remove_trivial_axes,
)
from .progress_logger import (
    ConsoleProgressLogger,
    NapariProgressLogger,
    ProgressLogger,
)
from .qt_progress_logger import QtProgressLogger
from .slice_processor import SliceProcessor

__all__ = [
    # image_util
    "IMAGE_EXTENSIONS",
    "collect_all_image_names",
    "create_artifact_name",
    "create_empty_instance_image",
    "get_axis_info",
    "get_axis_info_from_shape",
    "get_current_slice_indices",
    "get_ndim",
    "get_supported_axes_from_shape",
    "load_images_from_directory",
    "pad_to_largest",
    "remove_trivial_axes",
    # progress loggers
    "ProgressLogger",
    "NapariProgressLogger",
    "ConsoleProgressLogger",
    "QtProgressLogger",
    # dl_util
    "normalize_intensity",
    "compute_percentiles",
    "normalize_percentile",
    "collect_training_data",
    "divide_training_data",
    # slice_processor
    "SliceProcessor",
]
