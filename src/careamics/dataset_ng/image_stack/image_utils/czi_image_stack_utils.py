"""CZI image stack utilities."""


def are_czi_axes_valid(axes: str) -> bool:
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
