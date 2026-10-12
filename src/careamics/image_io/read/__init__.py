"""Functions relating to reading image files of different formats."""

__all__ = [
    "ReadFunc",
    "get_read_func",
    "read_tiff",
    "read_zarr",
]

from .get_func import ReadFunc, get_read_func
from .tiff import read_tiff
