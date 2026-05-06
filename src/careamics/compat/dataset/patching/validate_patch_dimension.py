"""Patch validation functions."""

from typing import Union

import numpy as np


def validate_patch_dimensions(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    is_3d_patch: bool,
) -> None:
    """
    Check patch size and array compatibility.

    This method validates the patch sizes with respect to the array dimensions:

    - Patch must have two dimensions fewer than the array (S and C).
    - Patch sizes are smaller than the corresponding array dimensions.

    If one of these conditions is not met, a ValueError is raised.

    This method should be called after inputs have been resized.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_size : Union[list[int], tuple[int, ...]]
        Size of the patches along each dimension of the array, except the first.
    is_3d_patch : bool
        Whether the patch is 3D or not.

    Raises
    ------
    ValueError
        If the patch size is not consistent with the array shape (one more array
        dimension).
    ValueError
        If the patch size in Z is larger than the array dimension.
    ValueError
        If either of the patch sizes in X or Y is larger than the corresponding array
        dimension.
    """
    if len(patch_size) != len(arr.shape[2:]):
        raise ValueError(
            f"There must be a patch size for each spatial dimensions "
            f"(got {patch_size} patches for dims {arr.shape}). Check the axes order."
        )

    # Sanity checks on patch sizes versus array dimension
    if is_3d_patch and patch_size[0] > arr.shape[-3]:
        raise ValueError(
            f"Z patch size is inconsistent with image shape "
            f"(got {patch_size[0]} patches for dim {arr.shape[1]}). Check the axes "
            f"order."
        )

    if patch_size[-2] > arr.shape[-2] or patch_size[-1] > arr.shape[-1]:
        raise ValueError(
            f"At least one of YX patch dimensions is larger than the corresponding "
            f"image dimension (got {patch_size} patches for dims {arr.shape[-2:]}). "
            f"Check the axes order."
        )
