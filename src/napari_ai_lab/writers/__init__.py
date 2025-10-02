"""
Label writers for different storage formats.

This package provides a flexible architecture for saving and loading labels
in various formats (numpy, zarr, tiff, etc.) with a common interface.
"""

from .base_writer import BaseWriter
from .numpy_writer import NumpyWriter

# Export available writers for easy selection
__all__ = ["BaseWriter", "NumpyWriter"]

# Registry of available writers for easy access
AVAILABLE_WRITERS = {
    "numpy": NumpyWriter,
    # Future writers can be added here:
    # "zarr": ZarrWriter,
    # "tiff": TiffWriter,
}


def get_writer(writer_type: str, **kwargs) -> BaseWriter:
    """
    Factory function to create a writer instance.

    Args:
        writer_type: Type of writer ("numpy", "zarr", "tiff", etc.)
        **kwargs: Additional arguments passed to the writer constructor

    Returns:
        Writer instance

    Raises:
        ValueError: If writer_type is not available
    """
    if writer_type not in AVAILABLE_WRITERS:
        available = ", ".join(AVAILABLE_WRITERS.keys())
        raise ValueError(
            f"Writer '{writer_type}' not available. Available writers: {available}"
        )

    return AVAILABLE_WRITERS[writer_type](**kwargs)
