"""StructN2V axes supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedStructAxis(str, BaseEnum):
    """Supported structN2V mask axes.

    Attributes
    ----------
    HORIZONTAL : str
        Horizontal axis.
    VERTICAL : str
        Vertical axis.
    NONE : str
        No axis, the mask is not applied.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    NONE = "none"
