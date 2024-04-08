from enum import Enum, EnumMeta
from typing import Any


class _ContainerEnum(EnumMeta):
    def __contains__(cls, item: Any) -> bool:
        try:
            cls(item)
        except ValueError:
            return False
        return True

    @classmethod
    def has_value(cls, value: Any) -> bool:
        return value in cls._value2member_map_


class BaseEnum(Enum, metaclass=_ContainerEnum):
    """Base Enum class, allowing checking if a value is in the enum.

    Example
    -------
    >>> from careamics.utils.base_enum import BaseEnum
    >>> # Define a new enum
    >>> class BaseEnumExtension(BaseEnum):
    ...     VALUE = "value"
    >>> # Check if value is in the enum
    >>> "value" in BaseEnumExtension
    True
    """

    pass
