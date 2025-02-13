"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "Configuration",
    "algorithm_factory",
    "load_configuration",
    "save_configuration",
]

from .careamist import CAREamist
from .config import (
    Configuration,
    algorithm_factory,
    load_configuration,
    save_configuration,
)
