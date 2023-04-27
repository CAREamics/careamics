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


def compute_overlap_predict(
    arr: np.ndarray, patch_size: Tuple[int], overlap: Tuple[int]
) -> Tuple[int]:
    total_patches = [
        np.ceil(
            (arr.shape[i + 1] - overlap[i] // 2) / (patch_size[i] - overlap[i] // 2)
        ).astype(int)
        for i in range(len(patch_size))
    ]

    step = [
        np.ceil(
            (patch_size[i] * total_patches[i] - arr.shape[i + 1])
            / max(1, total_patches[i] - 1)
        ).astype(int)
        for i in range(len(patch_size))
    ]
    updated_overlap = [patch_size[i] - step[i] for i in range(len(patch_size))]
    return [1, *step], updated_overlap


def _compute_patch_steps(patch_size: Tuple[int], overlaps: Tuple[int]) -> Tuple[int]:
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
        min(patch_size[i] - overlaps[i], patch_size[i]) for i in range(len(patch_size))
    ]
    return tuple(steps)


def compute_view_windows(
    patch_sizes: Tuple[int], overlaps: Tuple[int]
) -> Tuple[Tuple[int]]:
    """Compute window shapes, overlaps and steps.

    Parameters
    ----------
    patch_sizes : Tuple[int]
        Size of the patches
    overlaps : Tuple[int]
        Overlap between patches

    Returns
    -------
    Tuple[Tuple[int]]
        Window shapes, overlaps and steps
    """
    window_shape = [p for p in patch_sizes if p is not None]
    window_overlap = [o for o in overlaps if o is not None]
    steps = _compute_patch_steps(window_shape, window_overlap)

    return tuple(window_shape), steps


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
