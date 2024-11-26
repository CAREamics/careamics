"""Utility functions for the CAREamics CLI."""

from typing import Tuple


def handle_2D_3D_callback(value: Tuple[int, int, int]) -> Tuple[int, ...]:
    """
    Callback for options that require 2D or 3D inputs.

    In the case of 2D, the 3rd element should be set to -1.

    Parameters
    ----------
    value : (int, int, int)
        Tile size value.

    Returns
    -------
    (int, int, int) | (int, int)
        If the last element in `value` is -1 the tuple is reduced to the first two
        values.
    """
    if value[2] == -1:
        return value[:2]
    return value
