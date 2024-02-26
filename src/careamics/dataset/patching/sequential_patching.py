
from typing import  Generator, List, Optional, Tuple, Union

import numpy as np
from skimage.util import view_as_windows

from ..dataset_utils import reshape_data
from .validate_patch_dimension import validate_patch_dimensions


def _compute_number_of_patches(
    arr: np.ndarray, patch_sizes: Union[List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Compute the number of patches that fit in each dimension.

    Array must have one dimension more than the patches (C dimension).

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    patch_sizes : Tuple[int]
        Size of the patches.

    Returns
    -------
    Tuple[int]
        Number of patches in each dimension.
    """
    try:
        n_patches = [
            np.ceil(arr.shape[i] / patch_sizes[i]).astype(int)
            for i in range(len(patch_sizes))
        ]
    except IndexError as e:
        raise(
            f"Patch size {patch_sizes} is not compatible with array shape {arr.shape}"
        ) from e
    return tuple(n_patches)


def _compute_overlap(
    arr: np.ndarray, patch_sizes: Union[List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Compute the overlap between patches in each dimension.

    Array must be of dimensions C(Z)YX, and patches must be of dimensions YX or ZYX.
    If the array dimensions are divisible by the patch sizes, then the overlap is 0.
    Otherwise, it is the result of the division rounded to the upper value.

    Parameters
    ----------
    arr : np.ndarray
        Input array 3 or 4 dimensions.
    patch_sizes : Tuple[int]
        Size of the patches.

    Returns
    -------
    Tuple[int]
        Overlap between patches in each dimension.
    """
    n_patches = _compute_number_of_patches(arr, patch_sizes)

    overlap = [
        np.ceil(
            np.clip(n_patches[i] * patch_sizes[i] - arr.shape[i], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def _compute_patch_steps(
    patch_sizes: Union[List[int], Tuple[int, ...]], overlaps: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Compute steps between patches.

    Parameters
    ----------
    patch_sizes : Tuple[int]
        Size of the patches.
    overlaps : Tuple[int]
        Overlap between patches.

    Returns
    -------
    Tuple[int]
        Steps between patches.
    """
    steps = [
        min(patch_sizes[i] - overlaps[i], patch_sizes[i])
        for i in range(len(patch_sizes))
    ]
    return tuple(steps)


# TODO why stack the target here and not on a different dimension before calling this function?
def _compute_reshaped_view(
    arr: np.ndarray,
    window_shape: Tuple[int, ...],
    step: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute reshaped views of an array, where views correspond to patches.

    Parameters
    ----------
    arr : np.ndarray
        Array from which the views are extracted.
    window_shape : Tuple[int]
        Shape of the views.
    step : Tuple[int]
        Steps between views.
    output_shape : Tuple[int]
        Shape of the output array.

    Returns
    -------
    np.ndarray
        Array with views dimension.
    """
    rng = np.random.default_rng()

    if target is not None:
        arr = np.stack([arr, target], axis=0)
        window_shape = (arr.shape[0], *window_shape)
        step = (arr.shape[0], *step)
        output_shape = (arr.shape[0], -1, arr.shape[2], *output_shape[2:])

    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    if target is not None:
        rng.shuffle(patches, axis=1)
    else:
        rng.shuffle(patches, axis=0)
    return patches


def extract_patches_sequential(
    arr: np.ndarray,
    axes: str,
    patch_size: Union[List[int], Tuple[int]],
    target: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, ...], None, None]:
    """
    Generate patches from an array in a sequential manner.

    Array dimensions should be C(Z)YX, where C can be a singleton dimension. The patches
    are generated sequentially and cover the whole array.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.
    patch_size : Tuple[int]
        Patch sizes in each dimension.

    Returns
    -------
    Generator[Tuple[np.ndarray, ...], None, None]
        Generator of patches.
    """
    is_3d_patch = len(patch_size) == 3

    # Reshape data to SCZYX
    arr, _ = reshape_data(arr, axes)

    # Patches sanity check and update
    patch_size = validate_patch_dimensions(arr, patch_size, is_3d_patch)

    # Compute overlap
    overlaps = _compute_overlap(arr=arr, patch_sizes=patch_size)

    # Create view window and overlaps
    window_steps = _compute_patch_steps(patch_sizes=patch_size, overlaps=overlaps)

    output_shape = [-1,] + patch_size[1:]

    # Generate a view of the input array containing pre-calculated number of patches
    # in each dimension with overlap.
    # Resulting array is resized to (n_patches, C, Z, Y, X) or (n_patches, C, Y, X)
    patches = _compute_reshaped_view(
        arr,
        window_shape=patch_size,
        step=window_steps,
        output_shape=output_shape,
        target=target,
    )
    if target is not None:
        return (
            patches[0, ...],
            patches[1, ...]
        )
    else:
        return patches, None