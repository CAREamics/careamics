"""Transforms supported by CAREamics."""

from enum import StrEnum


class SupportedAugmentation(StrEnum):
    """Augmentations supported by CAREamics."""

    XY_FLIP = "XYFlip"
    XY_RANDOM_ROTATE90 = "XYRandomRotate90"


class SupportedTransform(StrEnum):
    """Transforms supported by CAREamics."""

    N2V_MANIPULATE = "N2VManipulate"
