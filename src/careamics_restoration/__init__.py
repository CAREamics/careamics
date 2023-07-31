"""Main module."""


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics_restoration")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .engine import Engine as Engine
