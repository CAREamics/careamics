from enum import Enum, EnumMeta

class _ContainerEnum(EnumMeta):

    def __contains__(cls, item) -> bool:
        try:
            cls(item)
        except ValueError:
            return False
        return True    

    
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


class BaseEnum(Enum, metaclass=_ContainerEnum):
    """Base Enum class, allowing checking if a value is in the enum.

    >>> "value" in BaseEnumExtension
    """
    pass
