from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics_restoration")
except PackageNotFoundError:
    __version__ = "uninstalled"

# TODO See todo in config.data.py, are_axes_valid needs to be refactored
# somewhere else to avoid circular imports
from .engine import UnsupervisedEngine
