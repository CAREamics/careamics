"""Axes validation utilities."""

_AXES = "STCZYX"

def check_axes_validity_1d(axes: str) -> None:
    """
      Sanity check on axes.

    The constraints on the axes are the following:
    - must be a combination of 'STCZYX'
    - must not contain duplicates
    - must contain at least 2 contiguous axes: X and Y
    - must contain at most 4 axes
    - cannot contain both S and T axes

    Axes do not need to be in the order 'STCZYX', as this depends on the user data.    This function validates axes strings that include 1D spatial data.

    Parameters
    ----------
    axes : str
        Axes string to validate.

    Raises
    ------
    ValueError
        If the axes are invalid.
    """
    # Check for valid characters
    valid_chars = set("STCZYX")
    if not set(axes).issubset(valid_chars):
        invalid_chars = set(axes) - valid_chars
        raise ValueError(
            f"Invalid axis characters: {invalid_chars}. "
            f"Valid characters are: {valid_chars}"
        )

    # Check for duplicates
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axes found in '{axes}'")

    # Check length limits
    if len(axes) < 1 or len(axes) > 6:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least 1 and at most 6 axes."
        )

    # Count spatial axes
    spatial_axes = [ax for ax in axes if ax in "XYZ"]

    # Must have at least one spatial axis
    if len(spatial_axes) == 0:
        raise ValueError(f"Axes must contain at least one spatial axis (X, Y, or Z)")

    # Check for both S and T (not allowed)
    if "S" in axes and "T" in axes:
        raise ValueError("Axes cannot contain both 'S' and 'T'")


def check_axes_validity(axes: str) -> None:
    """
    Check if the axes are valid.

    This function validates that axes are a combination of allowed characters,
    without duplicates, with at least 2 spatial dimensions, and following
    specific constraints.

    The axes must:
    - be a combination of 'STCZYX'
    - not contain duplicates
    - contain at least 2 contiguous axes: X and Y
    - contain at most 6 axes
    - not contain both S and T axes

    Parameters
    ----------
    axes : str
        Axes string to validate.

    Raises
    ------
    ValueError
        If the axes are invalid.
    """
    # First check with 1D validation for basic cases
    check_axes_validity_1d(axes)

    # Additional checks for 2D+ data
    spatial_axes = [ax for ax in axes if ax in "XYZ"]

    # For 2D+, require at least X and Y
    if len(spatial_axes) >= 2:
        if "X" not in axes or "Y" not in axes:
            raise ValueError("2D+ data must contain both 'X' and 'Y' axes")


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
