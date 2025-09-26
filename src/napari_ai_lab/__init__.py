try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)
from ._writer import write_multiple, write_single_image

__all__ = (
    "write_single_image",
    "write_multiple",
    "ExampleQWidget",
    "ImageThreshold",
    "threshold_autogenerate_widget",
    "threshold_magic_widget",
)
