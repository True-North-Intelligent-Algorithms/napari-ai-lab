"""
Artifact I/O for different storage formats.

This package provides a flexible architecture for saving and loading artifacts
in various formats (numpy, tiff, etc.) with a common interface.
"""

from .base_artifact_io import BaseArtifactIO
from .numpy_artifact_io import NumpyArtifactIO
from .stacked_sequence_artifact_io import StackedSequenceArtifactIO
from .tiff_artifact_io import TiffArtifactIO

# Export available Artifact I/O classes for easy selection
__all__ = [
    "BaseArtifactIO",
    "NumpyArtifactIO",
    "TiffArtifactIO",
    "StackedSequenceArtifactIO",
]

# Registry of available Artifact I/O implementations for easy access
AVAILABLE_ARTIFACT_IO = {
    "numpy": NumpyArtifactIO,
    "tiff": TiffArtifactIO,
    "stacked_sequence": StackedSequenceArtifactIO,
}


def get_artifact_io(io_type: str, **kwargs) -> BaseArtifactIO:
    """
    Factory function to create an Artifact I/O instance.

    Args:
        io_type: Type of Artifact I/O ("numpy", "tiff", etc.)
        **kwargs: Additional arguments passed to the Artifact I/O constructor

    Returns:
        Artifact I/O instance

    Raises:
        ValueError: If io_type is not available
    """
    if io_type not in AVAILABLE_ARTIFACT_IO:
        available = ", ".join(AVAILABLE_ARTIFACT_IO.keys())
        raise ValueError(
            f"Artifact I/O '{io_type}' not available. Available Artifact I/Os: {available}"
        )

    return AVAILABLE_ARTIFACT_IO[io_type](**kwargs)
