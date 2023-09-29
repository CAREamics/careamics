"""Main module."""


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = ["Engine", "Configuration", "load_configuration", "save_configuration"]

from .config import Configuration, load_configuration, save_configuration
from .engine import Engine as Engine
