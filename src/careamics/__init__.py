"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "CAREamistV2",
    "Configuration",
    "algorithm_factory",
    "disable_debug_logging",
    "enable_debug_logging",
]

from .careamist import CAREamist
from .careamist_v2 import CAREamistV2
from .config import Configuration, algorithm_factory
from .utils.logging import disable_debug_logging, enable_debug_logging
