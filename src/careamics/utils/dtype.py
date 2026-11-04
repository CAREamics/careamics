"""Utilities for converting data types."""

from collections.abc import Mapping
from typing import Any

import numpy as np


def convert_floats_to_float32(value: Any) -> Any:
    """Recursively convert floating-point values to NumPy float32 scalars.

    Mappings and sequences are copied, while values of other types are returned
    unchanged.

    Parameters
    ----------
    value : Any
        Value or nested collection to convert.

    Returns
    -------
    Any
        Converted value or collection.
    """
    if isinstance(value, Mapping):
        return {key: convert_floats_to_float32(item) for key, item in value.items()}

    if isinstance(value, list):
        return [convert_floats_to_float32(item) for item in value]

    if isinstance(value, tuple):
        return tuple(convert_floats_to_float32(item) for item in value)

    if isinstance(value, float | np.floating):
        return np.float32(value)

    return value
