"""
CellCast StarDist Global Segmenter.

This module provides a CellCast StarDist segmenter for automatic segmentation
of entire 2D images without user prompts using the CellCast versatile_fluo model.
"""

from dataclasses import dataclass

import numpy as np

from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import cellcast at module level. 0.3 replaced the stardist_2d
# functions with a StarDist2D model class.
try:
    import cellcast

    StarDist2D = cellcast.models.StarDist2D
    _is_cellcast_available = True
except (ImportError, AttributeError):
    StarDist2D = None
    _is_cellcast_available = False


@dataclass
class CellCastStardistSegmenter(GlobalSegmenterBase):
    """
    CellCast StarDist global segmenter

    Uses CellCast's versatile_fluo model to segment entire images automatically.
    """

    instructions = """
CellCast StarDist Automatic Segmentation:
• Automatically segments cells/objects in the entire image
• Uses CellCast's versatile_fluo model
• Optimized for fluorescence microscopy images
• GPU acceleration supported when available
    """

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        super().__init__()
        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        # Built on first use: init_fluo loads and pre-warms the model.
        self._model = None

    def are_dependencies_available(self) -> bool:
        """Check if CellCast is available."""
        return _is_cellcast_available

    def segment(self, image, **kwargs):
        """
        Perform segmentation using CellCast StarDist.

        Args:
            image: Input image (2D grayscale or RGB)
            **kwargs: Additional keyword arguments

        Returns:
            Label image with instance segmentations
        """
        if not _is_cellcast_available:
            raise ImportError(
                'CellCast is not available. Please install it with: pip install "cellcast>=0.3"'
            )

        if self._model is None:
            self._model = StarDist2D.init_fluo(gpu=True)

        labels = self._model.predict_fluo(image)

        print(f"✅ CellCast: Found {labels.max()} objects")

        return labels.astype(np.uint16)

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework(
            "CellCastStardistSegmenter", cls
        )
