"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "ImageStackLoading",
    "IndependentTargets",
    "NoiseModelTrainer",
    "ReadFuncLoading",
]

from .careamist import CAREamist
from .dataset.factory import ImageStackLoading, IndependentTargets, ReadFuncLoading
from .noise_model import NoiseModelTrainer
