"""Coordinate and patch filters supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedPatchFilters(str, BaseEnum):
    """Supported patch filters."""

    MAX = "max"
    MEANSTD = "mean_std"
    SHANNON = "shannon"


class SupportedCoordinateFilters(str, BaseEnum):
    """Supported coordinate filters."""

    MASK = "mask"
