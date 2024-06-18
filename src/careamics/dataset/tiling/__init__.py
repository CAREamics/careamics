"""Tiling functions."""

__all__ = [
    "stitch_prediction",
    "extract_tiles",
    "collate_tiles",
]

from .collate_tiles import collate_tiles
from .tiled_patching import extract_tiles
