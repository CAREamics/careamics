"""CAREamics"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics")
except PackageNotFoundError:
    __version__ = "uninstalled"
