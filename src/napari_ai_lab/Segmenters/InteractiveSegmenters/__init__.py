"""
Interactive Segmenters package.

This package provides a framework for interactive segmentation tools
that can be used with napari and other image analysis workflows.

IMPORTANT: To avoid dependency issues, import segmenters directly:
    from napari_ai_lab.Segmenters.InteractiveSegmenters.Square2D import Square2D
    from napari_ai_lab.Segmenters.InteractiveSegmenters.SAM3D import SAM3D  # if SAM deps available

Only the base class is imported by default. Individual segmenters should be
imported explicitly based on your environment and available dependencies.
"""

# Always import the base class
from .InteractiveSegmenterBase import InteractiveSegmenterBase

# List of available segmenters - only import on demand to avoid dependency issues
__all__ = ["InteractiveSegmenterBase"]

# Optional segmenters - only imported if their dependencies are available
_OPTIONAL_SEGMENTERS = {
    "Otsu2D": ".Otsu2D",
    "Otsu3D": ".Otsu3D",
    "Square2D": ".Square2D",
    "SAM3D": ".SAM3D",  # This one has heavy dependencies
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
