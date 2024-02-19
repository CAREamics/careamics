"""
Extraction strategy module.

This module defines the various extraction strategies available in CAREamics.
"""
from enum import Enum


class SupportedExtractionStrategy(str, Enum):
    """
    Available extraction strategies.

    Currently supported:
        - random: random extraction.
        # TODO
        - sequential: grid extraction, can miss edge values.
        - tiled: tiled extraction, covers the whole image.
    """

    RANDOM = "random"
    RANDOM_ZARR = "random_zarr"
    SEQUENTIAL = "sequential"
    TILED = "tiled"
