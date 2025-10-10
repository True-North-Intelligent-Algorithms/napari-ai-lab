"""
Base class for interactive segmenters with registry system.

This module provides a base class that interactive segmenters can inherit from,
along with a registry system for automatic discovery and registration of
segmenter implementations.
"""

from ..SegmenterBase import SegmenterBase


class InteractiveSegmenterBase(SegmenterBase):
    """
    Base class for interactive segmenters with registry functionality.

    This class provides interactive segmentation capabilities where segmenters
    use user prompts (points/shapes) to guide the segmentation process.

    Example usage:
        # In a derived segmenter class (e.g., CellPoseSegmenter):
        from .InteractiveSegmenterBase import InteractiveSegmenterBase

        class CellPoseSegmenter(InteractiveSegmenterBase):
            # Implementation here
            pass

        # Register the segmenter when the module is imported:
        InteractiveSegmenterBase.register_framework('CellPoseSegmenter', CellPoseSegmenter)
    """

    # Registry for interactive segmenter frameworks - separate from global segmenters
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add an interactive framework to the registry.

        Args:
            name (str): The name of the framework.
            framework (InteractiveSegmenterBase): The framework class to register.
        """
        cls.registry[name] = framework
        print(f"Registered interactive segmenter: {name}")

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform interactive segmentation on the given image.

        This is the main method that should be overridden by derived classes.
        Interactive segmenters use points and shapes to guide segmentation.

        Args:
            image (numpy.ndarray): Input image to segment.
            points (list, optional): List of annotation points for guidance.
            shapes (list, optional): List of annotation shapes for guidance.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Segmentation mask.

        Raises:
            NotImplementedError: If not implemented by derived class.
        """
        raise NotImplementedError(
            "Derived interactive segmenter classes must implement the segment method"
        )
