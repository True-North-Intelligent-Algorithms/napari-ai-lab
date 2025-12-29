"""
io for different storage formats.

This package provides a flexible architecture for saving and loading
in various formats (numpy, zarr, tiff, etc.) with a common interface.
"""

from .base_io import BaseIO
from .numpy_io import NumpyIO
from .stacked_sequence_io import StackedSequenceIO
from .tiff_io import TiffIO

# Export available I/O classes for easy selection
__all__ = ["BaseIO", "NumpyIO", "TiffIO", "StackedSequenceIO"]

# Registry of available I/O implementations for easy access
AVAILABLE_IO = {
    "numpy": NumpyIO,
    "tiff": TiffIO,
    "stacked_sequence": StackedSequenceIO,
}


def get_io(io_type: str, **kwargs) -> BaseIO:
    """
    Factory function to create a I/O instance.

    Args:
        io_type: Type of I/O ("numpy", "zarr", "tiff", etc.)
        **kwargs: Additional arguments passed to the I/O constructor

    Returns:
        I/O instance

    Raises:
        ValueError: If io_type is not available
    """
    if io_type not in AVAILABLE_IO:
        available = ", ".join(AVAILABLE_IO.keys())
        raise ValueError(
            f"I/O '{io_type}' not available. Available I/Os: {available}"
        )

    return AVAILABLE_IO[io_type](**kwargs)
