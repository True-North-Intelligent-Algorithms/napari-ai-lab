"""
Global Segmenters package.

This package provides a framework for global segmentation tools
that perform automatic segmentation on entire images without requiring
user prompts (points/shapes).

IMPORTANT: To avoid dependency issues, import segmenters directly:
    from napari_ai_lab.Segmenters.GlobalSegmenters.OtsuSegmenter import OtsuSegmenter
    from napari_ai_lab.Segmenters.GlobalSegmenters.ThresholdSegmenter import ThresholdSegmenter

Only the base class is imported by default. Individual segmenters should be
imported explicitly based on your environment and available dependencies.
"""

# Always import the base class
from .GlobalSegmenterBase import GlobalSegmenterBase

# List of available segmenters - only import on demand to avoid dependency issues
__all__ = ["GlobalSegmenterBase", "MicroSamSegmenter"]

# Optional segmenters - only imported if their dependencies are available
_OPTIONAL_SEGMENTERS = {
    "CellposeSegmenter": ".CellposeSegmenter",
    "StardistSegmenter": ".StardistSegmenter",
    "MicroSamSegmenter": ".MicroSamSegmenter",
    "MicroSamYoloSegmenter": ".MicroSamYoloSegmenter",
    "OtsuSegmenter": ".OtsuSegmenter",
    "ThresholdSegmenter": ".ThresholdSegmenter",
}


def _try_import_segmenter(name, module_path):
    """Try to import a segmenter, return None if dependencies missing."""
    try:
        from importlib import import_module

        module = import_module(module_path, package=__name__)
        return getattr(module, name)
    except ImportError as e:
        print(f"Warning: Could not import {name}: {e}")
        return None


# Make segmenters available as attributes but don't import them automatically
def __getattr__(name):
    """Lazy import segmenters to avoid dependency issues."""
    if name in _OPTIONAL_SEGMENTERS:
        segmenter_class = _try_import_segmenter(
            name, _OPTIONAL_SEGMENTERS[name]
        )
        if segmenter_class is not None:
            # Cache the successful import
            globals()[name] = segmenter_class
            return segmenter_class
        else:
            raise ImportError(f"Could not import {name} - check dependencies")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
