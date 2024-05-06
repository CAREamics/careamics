"""
Extraction strategy module.

This module defines the various extraction strategies available in CAREamics.
"""

from careamics.utils import BaseEnum


class SupportedExtractionStrategy(str, BaseEnum):
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
    NONE = "none"
