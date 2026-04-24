"""Functions relating reading and writing image files."""

__all__ = [
    "ReadFunc",
    "SupportedWriteType",
    "WriteFunc",
    "get_read_func",
    "get_write_func",
    "read",
    "write",
]

from . import read, write
from .read import ReadFunc, get_read_func
from .write import SupportedWriteType, WriteFunc, get_write_func
