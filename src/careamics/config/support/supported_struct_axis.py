"""StructN2V axes supported by CAREamics."""

from enum import StrEnum


class SupportedStructAxis(StrEnum):
    """Supported structN2V mask axes.

    Attributes
    ----------
    HORIZONTAL : str
        Horizontal axis.
    VERTICAL : str
        Vertical axis.
    CROSS : str
        Both axes, i.e. horizontal and vertical.
    SQUARE : str
        Square mask, i.e. horizontal and vertical axes.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CROSS = "cross"
    SQUARE = "square"
