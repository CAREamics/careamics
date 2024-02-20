from typing import List, Tuple, Union

import numpy as np


def validate_patch_dimensions(
    arr: np.ndarray,
    patch_size: Union[List[int], Tuple[int, ...]],
    is_3d_patch: bool,
) -> Tuple[int, ...]:
    """
    Check patch size and array compatibility.

    This method validates the patch sizes with respect to the array dimensions:
    - The patch sizes must have one dimension fewer than the array (C dimension).
    - Chack that patch sizes are smaller than array dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_size : Union[List[int], Tuple[int, ...]]
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
            f"(got {patch_size} patches for dims {arr.shape})."
        )

    # Sanity checks on patch sizes versus array dimension
    if is_3d_patch and patch_size[0] > arr.shape[-3]:
        raise ValueError(
            f"Z patch size is inconsistent with image shape "
            f"(got {patch_size[0]} patches for dim {arr.shape[1]})."
        )

    if patch_size[-2] > arr.shape[-2] or patch_size[-1] > arr.shape[-1]:
        raise ValueError(
            f"At least one of YX patch dimensions is larger than the corresponding "
            f"image dimension (got {patch_size} patches for dims {arr.shape[-2:]})."
        )
    
    # Update patch size to SC(Z)YX format
    return [1, arr.shape[1], *patch_size]