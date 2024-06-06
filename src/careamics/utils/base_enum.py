"""A base class for Enum that allows checking if a value is in the Enum."""

from enum import Enum, EnumMeta
from typing import Any


class _ContainerEnum(EnumMeta):
    """Metaclass for Enum with __contains__ method."""

    def __contains__(cls, item: Any) -> bool:
        """Check if an item is in the Enum.

        Parameters
        ----------
        item : Any
            Item to check.

        Returns
        -------
        bool
            True if the item is in the Enum, False otherwise.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True

    @classmethod
    def has_value(cls, value: Any) -> bool:
        """Check if a value is in the Enum.

        Parameters
        ----------
        value : Any
            Value to check.

        Returns
        -------
        bool
            True if the value is in the Enum, False otherwise.
        """
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
