from careamics.utils import BaseEnum


class SupportedStructAxis(str, BaseEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    NONE = "none"
