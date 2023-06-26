from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("careamics_restoration")
except PackageNotFoundError:
    __version__ = "uninstalled"

# TODO this leads to circular imports from fe13743, investigate
# from .engine import UnsupervisedEngine
