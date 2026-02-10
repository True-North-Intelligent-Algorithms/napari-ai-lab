try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .apps import NDEasyLabel, NDEasySegment

__all__ = ("NDEasyLabel", "NDEasySegment")
