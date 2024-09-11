"""Functions relating to writing image files of different formats."""

__all__ = [
    "get_write_func",
    "write_tiff",
    "WriteFunc",
    "SupportedWriteType",
]

from .get_func import (
    SupportedWriteType,
    WriteFunc,
    get_write_func,
)
from .tiff import write_tiff
