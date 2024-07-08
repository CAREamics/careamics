"""Functions relating to writing image files of different formats."""

__all__ = ["get_write_func", "write_tiff", "WriteFunc"]

from .get_func import WriteFunc, get_write_func
from .tiff import write_tiff
