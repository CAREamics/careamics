from careamics.utils.base_enum import BaseEnum


class MyEnum(str, BaseEnum):
    A = "a"
    B = "b"
    C = "c"


def test_base_enum():
    """Test that BaseEnum allows the `in` operator with values."""
    assert "b" in MyEnum
