"""Axes validation utilities."""

_AXES = "STCZYX"


def check_axes_validity(axes: str) -> None:
    """
    Sanity check on axes.

    The constraints on the axes are the following:
    - must be a combination of 'STCZYX'
    - must not contain duplicates
    - must contain at least 2 contiguous axes: X and Y
    - must contain at most 4 axes

    Axes do not need to be in the order 'STCZYX', as this depends on the user data.

    Parameters
    ----------
    axes : str
        Axes to validate.
    """
    _axes = axes.upper()

    # Minimum is 2 (XY) and maximum is 4 (TZYX)
    if len(_axes) < 2 or len(_axes) > 6:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least 2 and at most 6 axes."
        )

    if "YX" not in _axes and "XY" not in _axes:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least X and Y axes consecutively."
        )

    # all characters must be in REF_AXES = 'STCZYX'
    if not all(s in _AXES for s in _axes):
        raise ValueError(f"Invalid axes {axes}. Must be a combination of {_AXES}.")

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            raise ValueError(
                f"Invalid axes {axes}. Cannot contain duplicate axes"
                f" (got multiple {axes[i]})."
            )


def check_czi_axes_validity(axes: str) -> bool:
    """
    Check if the provided axes string is valid for CZI files.

    CZI axes is always in the "SC(Z/T)YX" format, where Z or T are optional, and S and C
    can be singleton dimensions, but must be provided.

    Parameters
    ----------
    axes : str
        The axes string to validate.

    Returns
    -------
    bool
        True if the axes string is valid, False otherwise.
    """
    valid_axes = {"S", "C", "Z", "T", "Y", "X"}
    axes_set = set(axes)

    # check for invalid characters
    if not axes_set.issubset(valid_axes):
        return False

    # check for mandatory axes
    if not ({"S", "C", "Y", "X"}.issubset(axes_set)):
        return False

    # check for mutually exclusive axes
    if "Z" in axes_set and "T" in axes_set:
        return False

    # check for correct order
    order = "SCZYX" if "Z" in axes else "SCTYX"
    last_index = -1
    for axis in axes:
        current_index = order.find(axis)
        if current_index < last_index:
            return False
        last_index = current_index

    return True
