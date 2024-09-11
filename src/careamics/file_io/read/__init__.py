"""Functions relating to reading image files of different formats."""

__all__ = [
    "get_read_func",
    "read_tiff",
    "read_zarr",
    "ReadFunc",
]

from .get_func import ReadFunc, get_read_func
from .tiff import read_tiff
from .zarr import read_zarr
