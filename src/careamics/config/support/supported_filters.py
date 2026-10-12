"""Coordinate and patch filters supported by CAREamics."""

from enum import StrEnum


class SupportedPatchFilters(StrEnum):
    """Supported patch filters."""

    MAX = "max"
    MEANSTD = "mean_std"
    SHANNON = "shannon"


class SupportedCoordinateFilters(StrEnum):
    """Supported coordinate filters."""

    MASK = "mask"
