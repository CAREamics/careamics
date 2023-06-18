from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("n2v")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .engine import UnsupervisedEngine
