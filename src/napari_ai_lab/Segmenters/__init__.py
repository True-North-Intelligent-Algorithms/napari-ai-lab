"""
Segmenters package.

This package provides a unified framework for different types of segmentation tools:
- SegmenterBase: Common base class with shared functionality
- InteractiveSegmenters: Segmenters that use user prompts (points/shapes)
- GlobalSegmenters: Segmenters that process entire images automatically

The package is organized to allow for future expansion and common base classes.

IMPORTANT: To avoid dependency issues, import specific segmenters directly:
    from napari_ai_lab.Segmenters.GlobalSegmenters.OtsuSegmenter import OtsuSegmenter
    from napari_ai_lab.Segmenters.InteractiveSegmenters.Square2D import Square2D

Only the base classes are imported by default. Individual segmenters should be
imported explicitly to avoid pulling in unwanted dependencies.
"""

# Import only the essential base classes by default
from .SegmenterBase import SegmenterBase

# Import base classes from subpackages
try:
    from .InteractiveSegmenters.InteractiveSegmenterBase import (
        InteractiveSegmenterBase,
    )
except ImportError as e:
    print(f"Warning: Could not import InteractiveSegmenterBase: {e}")
    InteractiveSegmenterBase = None

try:
    from .GlobalSegmenters.GlobalSegmenterBase import GlobalSegmenterBase
except ImportError as e:
    print(f"Warning: Could not import GlobalSegmenterBase: {e}")
    GlobalSegmenterBase = None

# Only export the base classes by default
# Users should import specific segmenters directly to avoid dependency issues
__all__ = [
    "SegmenterBase",
]

# Add base classes if they were successfully imported
if InteractiveSegmenterBase is not None:
    __all__.append("InteractiveSegmenterBase")
if GlobalSegmenterBase is not None:
    __all__.append("GlobalSegmenterBase")
