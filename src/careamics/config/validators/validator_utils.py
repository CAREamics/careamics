"""
Validator functions.

These functions are used to validate dimensions and axes of inputs.
"""
from typing import Tuple, Optional, List, Union

_AXES = "STCZYX"


def check_axes_validity(axes: str) -> bool:
    """
    Sanity check on axes.

    The constraints on the axes are the following:
    - must be a combination of 'STCZYX'
    - must not contain duplicates
    - must contain at least 2 contiguous axes: X and Y
    - must contain at most 4 axes
    - cannot contain both S and T axes

    Axes do not need to be in the order 'STCZYX', as this depends on the user data.

    Parameters
    ----------
    axes : str
        Axes to validate.

    Returns
    -------
    bool
        True if axes are valid, False otherwise.
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

    return True


def patch_size_ge_than_8_power_of_2(
        patch_list: Optional[Union[List[int], Tuple[int]]]
    ) -> Optional[Union[List[int], Tuple[int]]]:
    """
    Validate that each entry is greater or equal than 8 and a power of 2.

    If None is passed, the function will return None.

    Parameters
    ----------
    patch_list : Optional[Union[List[int], Tuple[int]]]
        Patch size.

    Returns
    -------
    Optional[Union[List[int], Tuple[int]]]
        Validated patch size.

    Raises
    ------
    ValueError
        If the patch size if smaller than 8.
    ValueError
        If the patch size is not a power of 2.
    """
    if patch_list is not None:
        for dim in patch_list:
            if dim < 8:
                raise ValueError(
                    f"Patch size must be non-zero positive (got {dim})."
                )

            if (dim & (dim - 1)) != 0:
                raise ValueError(
                    f"Patch size must be a power of 2 in all dimensions "
                    f"(got {dim})."
                )
            
    return patch_list