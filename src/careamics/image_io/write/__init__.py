"""Functions relating to writing image files of different formats."""

__all__ = [
    "SupportedWriteType",
    "WriteFunc",
    "get_write_func",
    "write_tiff",
]

from .get_func import (
    SupportedWriteType,
    WriteFunc,
    get_write_func,
)
from .tiff import write_tiff
