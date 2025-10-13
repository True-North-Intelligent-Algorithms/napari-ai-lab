"""
Square2D Interactive Segmenter.

This module provides a simple 2D square segmentation tool that creates
square regions around points for annotation purposes.
"""

import dataclasses

import numpy as np

from .InteractiveSegmenterBase import InteractiveSegmenterBase


@dataclasses.dataclass
class Square2D(InteractiveSegmenterBase):
    """
    2D Square Interactive Segmenter.

    Creates square regions around points for quick annotation.
    """

    # Instructions displayed in the UI
    instructions = """
    0. This segmenter is for testing purposes only.
    1. Select Points Layer
    2. Click to add points
    3. At each point a square label should appear
    4. Adjust size parameter to control square dimensions."
    """

    # Parameters with metadata for automatic UI generation
    size: int = dataclasses.field(
        default=20,
        metadata={
            "type": "int",
            "min": 1,
            "max": 200,
            "step": 1,
            "default": 20,
        },
    )

    def __init__(self, name=None):
        """
        Initialize the SAM3D segmenter.

        Args:
            name (str, optional): Name of this segmenter instance.
        """
        super().__init__(name)

    @property
    def supported_axes(self) -> list[str]:
        """Return the supported axes for this segmenter."""
        return [
            "YX",
            "ZYX",
            "TYX",
            "TZYX",
        ]  # Supports 2D with optional leading dimensions

    def segment(
        self,
        image: np.ndarray,
        points: list[tuple[float, ...]] | None = None,
        shapes: list[np.ndarray] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform square segmentation around provided points.

        Parameters
        ----------
        image : np.ndarray
            Input image array (any dimensionality, last 2 dims are Y, X).
        points : List[Tuple[float, ...]], optional
            List of points where each point has coordinates matching image dimensions.
            For 2D: (y, x), for 3D: (z, y, x), etc.
        shapes : List[np.ndarray], optional
            Not used for Square2D segmentation.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Binary mask with same shape as input image, True where squares are drawn.
        """
        if image is None:
            raise ValueError("Image cannot be None")

        # Initialize empty mask
        mask = np.zeros(image.shape, dtype=bool)

        if points is None or len(points) == 0:
            return mask

        # Process each point
        for point in points:
            self._create_square_at_point(mask, point)

        return mask

    def _create_square_at_point(
        self, mask: np.ndarray, point: tuple[float, ...]
    ) -> None:
        """
        Create a square region in the mask around the specified point.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask to modify in-place.
        point : Tuple[float, ...]
            Point coordinates (last two are y, x).
        """
        # Extract coordinates
        *leading, y, x = point

        # Convert to integers
        leading = [int(coord) for coord in leading]
        y, x = int(y), int(x)

        half = self.size // 2

        # Calculate bounds (ensure within image boundaries)
        y0 = max(0, y - half)
        y1 = min(mask.shape[-2], y + half)
        x0 = max(0, x - half)
        x1 = min(mask.shape[-1], x + half)

        # Build slice object for all dimensions
        index = tuple(leading) + (slice(y0, y1), slice(x0, x1))

        # Set the square region to True in the mask
        mask[index] = True

    def get_parameters_dict(self) -> dict:
        """
        Get current parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary of parameter names to current values.
        """
        return {"size": self.size}

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return InteractiveSegmenterBase.register_framework("Square2D", cls)
