"""
Extraction strategy module.

This module defines the various extraction strategies available in CAREamics.
"""

from enum import Enum


class ExtractionStrategy(str, Enum):
    """
    Available extraction strategies.

    Currently supported:
        - random: random extraction.
        - sequential: grid extraction, can miss edge values.
        - tiled: tiled extraction, covers the whole image.
    """

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    TILED = "tiled"
