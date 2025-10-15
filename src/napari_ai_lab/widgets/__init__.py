"""
Widgets module for napari-ai-lab.

This module contains custom Qt widgets for the napari-ai-lab plugin.
"""

from .parameter_slider import ParameterSlider
from .segmenter_widget import SegmenterWidget

__all__ = ["SegmenterWidget", "ParameterSlider"]
