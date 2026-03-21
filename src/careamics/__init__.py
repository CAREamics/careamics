"""Main CAREamics module."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "CAREamist",
    "Configuration",
    "NoiseModelTrainer",
    "algorithm_factory",
]

from .careamist import CAREamist
from .config import (
    Configuration,
    algorithm_factory,
)
from .noise_model import NoiseModelTrainer
