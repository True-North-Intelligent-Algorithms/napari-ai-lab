"""
Base class for global segmenters with registry system.

This module provides a base class that global segmenters can inherit from,
along with a registry system for automatic discovery and registration of
segmenter implementations.

Global segmenters perform automatic segmentation on entire images without
requiring user prompts (points/shapes).
"""

from ..SegmenterBase import SegmenterBase


class GlobalSegmenterBase(SegmenterBase):
    """
    Base class for global segmenters with registry functionality.

    This class provides global segmentation capabilities where segmenters
    process entire images automatically without requiring user prompts.

    Global segmenters are designed for automatic segmentation workflows
    where no user interaction (points/shapes) is required and the entire
    image is processed globally.

    Example usage:
        # In a derived segmenter class (e.g., OtsuSegmenter):
        from .GlobalSegmenterBase import GlobalSegmenterBase

        class OtsuSegmenter(GlobalSegmenterBase):
            # Implementation here
            pass

        # Register the segmenter when the module is imported:
        GlobalSegmenterBase.register_framework('OtsuSegmenter', OtsuSegmenter)
    """

    # Registry for global segmenter frameworks - separate from interactive segmenters
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add a global framework to the registry.

        Args:
            name (str): The name of the framework.
            framework (GlobalSegmenterBase): The framework class to register.
        """
        cls.registry[name] = framework
        print(f"Registered global segmenter: {name}")

    def segment(self, image, **kwargs):
        """
        Perform global segmentation on the given image.

        This is the main method that should be overridden by derived classes.
        Global segmenters work on the entire image without user prompts.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments specific to the segmenter.

        Returns:
            numpy.ndarray: Segmentation mask or labeled image.

        Raises:
            NotImplementedError: If not implemented by derived class.
        """
        raise NotImplementedError(
            "Derived global segmenter classes must implement the segment method"
        )
