"""
Validator functions.

These functions are used to validate dimensions and axes of inputs.
"""

from collections.abc import Sequence


def _value_ge_than_8_power_of_2(
    value: int,
) -> None:
    """
    Validate that the value is greater or equal than 8 and a power of 2.

    Parameters
    ----------
    value : int
        Value to validate.

    Raises
    ------
    ValueError
        If the value is smaller than 8.
    ValueError
        If the value is not a power of 2.
    """
    if value < 8:
        raise ValueError(f"Value must be greater than 8 (got {value}).")

    if (value & (value - 1)) != 0:
        raise ValueError(f"Value must be a power of 2 (got {value}).")


def patch_size_ge_than_8_power_of_2(
    patch_list: Sequence[int] | None,
) -> None:
    """
    Validate that each entry is greater or equal than 8 and a power of 2.

    Parameters
    ----------
    patch_list : Sequence of int, or None
        Patch size.

    Raises
    ------
    ValueError
        If the patch size if smaller than 8.
    ValueError
        If the patch size is not a power of 2.
    """
    if patch_list is not None:
        for dim in patch_list:
            _value_ge_than_8_power_of_2(dim)
