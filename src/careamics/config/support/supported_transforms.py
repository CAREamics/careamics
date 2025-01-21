"""Transforms supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedTransform(str, BaseEnum):
    """Transforms officially supported by CAREamics."""

    XY_FLIP = "XYFlip"
    XY_RANDOM_ROTATE90 = "XYRandomRotate90"
    NORMALIZE = "Normalize"
    N2V_MANIPULATE = "N2VManipulate"
