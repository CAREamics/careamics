"""Functions relating reading and writing image files."""

__all__ = [
    "read",
    "write",
    "get_read_func",
    "get_write_func",
    "ReadFunc",
    "WriteFunc",
]

from . import read, write
from .read import ReadFunc, get_read_func
from .write import WriteFunc, get_write_func
