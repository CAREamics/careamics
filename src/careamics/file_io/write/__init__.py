"""Functions relating to writing image files of different formats."""

__all__ = [
    "get_write_func",
    "write_tiff",
    "WriteFunc"
]

from .get_func import get_write_func, WriteFunc
from .tiff import write_tiff
