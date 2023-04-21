from typing import Tuple

import numpy as np
from skimage.util import view_as_windows


def _compute_number_of_patches(arr: np.ndarray, patch_sizes: Tuple[int]) -> Tuple[int]:
    """Compute a number of patches in each dimension in order to covert the whole
    array.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patche_sizes : Tuple[int]
        Size of the patches

    Returns
    -------
    Tuple[int]
        Number of patches in each dimension
    """
    n_patches = [
        np.ceil(arr.shape[i + 1] / patch_sizes[i]).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(n_patches)


def compute_overlap(arr: np.ndarray, patch_sizes: Tuple[int]) -> Tuple[int]:
    """Compute the overlap between patches in each dimension.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.
    If the array dimensions are divisible by the patch sizes, then the overlap is 0.
    Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patche_sizes : Tuple[int]
        Size of the patches

    Returns
    -------
    Tuple[int]
        Overlap between patches in each dimension
    """
    n_patches = _compute_number_of_patches(arr, patch_sizes)

    overlap = [
        np.ceil(
            (n_patches[i] * patch_sizes[i] - arr.shape[1 + i])
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def compute_patch_steps(patch_sizes: Tuple[int], overlaps: Tuple[int]) -> Tuple[int]:
    """Compute steps between patches.

    Parameters
    ----------
    patch_size : Tuple[int]
        Size of the patches
    overlaps : Tuple[int]
        Overlap between patches

    Returns
    -------
    Tuple[int]
        Steps between patches
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


def compute_reshaped_view(
    arr: np.ndarray,
    window_shape: Tuple[int],
    step: Tuple[int],
    output_shape: Tuple[int],
) -> np.ndarray:
    """Compute the reshaped views of an array.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted
    window_shape : Tuple[int]
        Shape of the views
    step : Tuple[int]
        Steps between views
    output_shape : Tuple[int]
        Shape of the output array
    """
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )

    return patches
