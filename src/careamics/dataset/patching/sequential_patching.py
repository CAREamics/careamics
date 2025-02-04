"""Sequential patching functions."""

from typing import Optional, Union

import numpy as np
from skimage.util import view_as_windows

from .validate_patch_dimension import validate_patch_dimensions


def _compute_number_of_patches(
    arr_shape: tuple[int, ...], patch_sizes: Union[list[int], tuple[int, ...]]
) -> tuple[int, ...]:
    """
    Compute the number of patches that fit in each dimension.

    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Shape of the input array.
    patch_sizes : Union[list[int], tuple[int, ...]
        Shape of the patches.

    Returns
    -------
    tuple[int, ...]
        Number of patches in each dimension.
    """
    if len(arr_shape) != len(patch_sizes):
        raise ValueError(
            f"Array shape {arr_shape} and patch size {patch_sizes} should have the "
            f"same dimension, including singleton dimension for S and equal dimension "
            f"for C."
        )

    try:
        n_patches = [
            np.ceil(arr_shape[i] / patch_sizes[i]).astype(int)
            for i in range(len(patch_sizes))
        ]
    except IndexError as e:
        raise ValueError(
            f"Patch size {patch_sizes} is not compatible with array shape {arr_shape}"
        ) from e

    return tuple(n_patches)


def _compute_overlap(
    arr_shape: tuple[int, ...], patch_sizes: Union[list[int], tuple[int, ...]]
) -> tuple[int, ...]:
    """
    Compute the overlap between patches in each dimension.

    If the array dimensions are divisible by the patch sizes, then the overlap is
    0. Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr_shape : tuple[int, ...]
        Input array shape.
    patch_sizes : Union[list[int], tuple[int, ...]]
        Size of the patches.

    Returns
    -------
    tuple[int, ...]
        Overlap between patches in each dimension.
    """
    n_patches = _compute_number_of_patches(arr_shape, patch_sizes)

    overlap = [
        np.ceil(
            np.clip(n_patches[i] * patch_sizes[i] - arr_shape[i], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def _compute_patch_steps(
    patch_sizes: Union[list[int], tuple[int, ...]], overlaps: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Compute steps between patches.

    Parameters
    ----------
    patch_sizes : tuple[int]
        Size of the patches.
    overlaps : tuple[int]
        Overlap between patches.

    Returns
    -------
    tuple[int]
        Steps between patches.
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


# TODO why stack the target here and not on a different dimension before this function?
def _compute_patch_views(
    arr: np.ndarray,
    window_shape: list[int],
    step: tuple[int, ...],
    output_shape: list[int],
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute views of an array corresponding to patches.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted.
    window_shape : tuple[int]
        Shape of the views.
    step : tuple[int]
        Steps between views.
    output_shape : tuple[int]
        Shape of the output array.
    target : Optional[np.ndarray], optional
        Target array, by default None.

    Returns
    -------
    np.ndarray
        Array with views dimension.
    """
    rng = np.random.default_rng()

    if target is not None:
        arr = np.stack([arr, target], axis=0)
        window_shape = [arr.shape[0], *window_shape]
        step = (arr.shape[0], *step)
        output_shape = [-1, arr.shape[0], arr.shape[2], *output_shape[2:]]

    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches


def extract_patches_sequential(
    arr: np.ndarray,
    patch_size: Union[list[int], tuple[int, ...]],
    target: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate patches from an array in a sequential manner.

    Array dimensions should be SC(Z)YX, where S and C can be singleton dimensions. The
    patches are generated sequentially and cover the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : tuple[int]
        Patch sizes in each dimension.
    target : Optional[np.ndarray], optional
        Target array, by default None.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray]]
        Patches.
    """
    is_3d_patch = len(patch_size) == 3

    # Patches sanity check
    validate_patch_dimensions(arr, patch_size, is_3d_patch)

    # Update patch size to encompass S and C dimensions
    patch_size = [1, arr.shape[1], *patch_size]

    # Compute overlap
    overlaps = _compute_overlap(arr_shape=arr.shape, patch_sizes=patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=patch_size, overlaps=overlaps)

    output_shape = [
        -1,
    ] + patch_size[1:]

    # Generate a view of the input array containing pre-calculated number of patches
    # in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches, C, Y, X)
    patches = _compute_patch_views(
        arr,
        window_shape=patch_size,
        step=window_steps,
        output_shape=output_shape,
        target=target,
    )

    if target is not None:
        # target was concatenated to patches in _compute_reshaped_view
        return (
            patches[:, 0, ...],
            patches[:, 1, ...],
        )  # TODO  in _compute_reshaped_view?
    else:
        return patches, None
