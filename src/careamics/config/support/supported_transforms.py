"""Transforms supported by CAREamics."""

from enum import StrEnum


class SupportedTransform(StrEnum):
    """Transforms officially supported by CAREamics."""

    XY_FLIP = "XYFlip"
    XY_RANDOM_ROTATE90 = "XYRandomRotate90"
    N2V_MANIPULATE = "N2VManipulate"
