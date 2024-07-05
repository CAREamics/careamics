"""Functions relating reading and writing image files."""

__all__ = ["read", "write", "get_read_func", "get_write_func"]

from . import read, write
from .read import get_read_func, ReadFunc
from .write import get_write_func, WriteFunc
