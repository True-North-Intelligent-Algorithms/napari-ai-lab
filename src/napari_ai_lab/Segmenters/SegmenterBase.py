"""
Base class for all segmenters with registry system.

This module provides a common base class that all segmenter types can inherit from,
containing shared functionality like registry management, axis support, and basic
segmenter operations.
"""


class SegmenterBase:
    """
    Common base class for all segmenters with registry functionality.

    This class provides shared functionality that is common to all types of
    segmenters including registry management, axis support, initialization,
    and string representations.

    This class should not be used directly - instead use InteractiveSegmenterBase
    or GlobalSegmenterBase depending on your segmentation approach.
    """

    # Registry for all segmenter frameworks - each subclass maintains its own
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add a framework to the registry.

        Args:
            name (str): The name of the framework.
            framework (SegmenterBase): The framework class to register.
        """
        cls.registry[name] = framework
        print(f"Registered segmenter: {name}")

    @classmethod
    def get_registered_frameworks(cls):
        """
        Get all registered frameworks for this segmenter type.

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
            SegmenterBase: The framework class, or None if not found.
        """
        return cls.registry.get(name)

    @classmethod
    def list_frameworks(cls):
        """
        List all registered framework names for this segmenter type.

        Returns:
            list: List of registered framework names.
        """
        return list(cls.registry.keys())

    def __init__(self):
        """Initialize the segmenter."""
        self.name = self.__class__.__name__

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

    def get_parameters_dict(self):
        """
        Get current parameters as a dictionary.

        This method should be overridden by derived classes that use dataclass
        parameters to return the current parameter values.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return {}

    def initialize_predictor(self, image, save_path: str, image_name: str):
        """
        Initialize the predictor for this segmenter.

        This is a base implementation that does nothing. Segmenters that require
        predictor initialization (like SAM models) should override this method.

        Args:
            image: Current image data for predictor initialization
            save_path (str): Directory path where embeddings/models are saved/loaded
            image_name (str): Name of the image (without extension)
        """

    def __str__(self):
        """String representation of the segmenter."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self):
        """Detailed string representation of the segmenter."""
        return f"{self.__class__.__name__}(name='{self.name}', registered_as={self.name in self.registry})"
