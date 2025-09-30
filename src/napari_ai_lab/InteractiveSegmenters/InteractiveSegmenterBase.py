"""
Base class for interactive segmenters with registry system.

This module provides a base class that interactive segmenters can inherit from,
along with a registry system for automatic discovery and registration of
segmenter implementations.
"""


class InteractiveSegmenterBase:
    """
    Base class for interactive segmenters with registry functionality.

    This class provides a registry system that allows segmenter frameworks
    to register themselves for automatic discovery and use.

    Example usage:
        # In a derived segmenter class (e.g., CellPoseSegmenter):
        from .InteractiveSegmenterBase import InteractiveSegmenterBase

        class CellPoseSegmenter(InteractiveSegmenterBase):
            # Implementation here
            pass

        # Register the segmenter when the module is imported:
        InteractiveSegmenterBase.register_framework('CellPoseSegmenter', CellPoseSegmenter)
    """

    # Initiate registry for all the frameworks. Each framework should register itself when imported:
    # For example a cellpose framework would have the following line in its code:
    # InteractiveSegmenterBase.register_framework('CellPoseInstanceFramework', CellPoseInstanceFramework)
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add a framework to the registry.

        Args:
            name (str): The name of the framework.
            framework (InteractiveSegmenterBase): The framework class to register.
        """
        cls.registry[name] = framework
        print(f"Registered interactive segmenter: {name}")

    @classmethod
    def get_registered_frameworks(cls):
        """
        Get all registered frameworks.

        Returns:
            dict: Dictionary of registered frameworks {name: framework_class}
        """
        return cls.registry.copy()

    @classmethod
    def get_framework(cls, name):
        """
        Get a specific framework by name.

        Args:
            name (str): The name of the framework to retrieve.

        Returns:
            InteractiveSegmenterBase: The framework class, or None if not found.
        """
        return cls.registry.get(name)

    @classmethod
    def list_frameworks(cls):
        """
        List all registered framework names.

        Returns:
            list: List of registered framework names.
        """
        return list(cls.registry.keys())

    def __init__(self, name=None):
        """
        Initialize the interactive segmenter.

        Args:
            name (str, optional): Name of this segmenter instance.
        """
        self.name = name or self.__class__.__name__

    @property
    def supported_axes(self):
        """
        Get the list of axis configurations this segmenter supports.

        Returns:
            list: List of supported axis strings (e.g., ['YX', 'YXC', 'ZYX']).
                 Empty list in base class - should be overridden by derived classes.
        """
        return []

    def supports_axis(self, axis_info):
        """
        Check if this segmenter supports the given axis configuration.

        Args:
            axis_info (str): Axis configuration string (e.g., 'YX', 'ZYX', 'YXC').

        Returns:
            bool: True if the axis configuration is supported, False otherwise.
        """
        return axis_info in self.supported_axes

    def segment(self, image, points=None, shapes=None, **kwargs):
        """
        Perform segmentation on the given image.

        This is a base method that should be overridden by derived classes.

        Args:
            image (numpy.ndarray): Input image to segment.
            points (list, optional): List of annotation points.
            shapes (list, optional): List of annotation shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Segmentation mask.

        Raises:
            NotImplementedError: If not implemented by derived class.
        """
        raise NotImplementedError(
            "Derived classes must implement the segment method"
        )

    def __str__(self):
        """String representation of the segmenter."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self):
        """Detailed string representation of the segmenter."""
        return f"{self.__class__.__name__}(name='{self.name}', registered_as={self.name in self.registry})"
