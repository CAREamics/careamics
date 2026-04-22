"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "ImageStackLoading",
    "ReadFuncLoading",
]

from .careamist import CAREamist
from .dataset.factory import ImageStackLoading, ReadFuncLoading
