from typing import Tuple

import numpy as np
from skimage.util import view_as_windows


AXES = "STCZYX"


def are_axes_valid(axes: str) -> bool:
    """Sanity check on axes.

    The constraints on the axes are the following:
    - must be a combination of 'STCZYX'
    - must not contain duplicates
    - must contain at least 2 contiguous axes: X and Y
    - must contain at most 4 axes
    - cannot contain both S and T axes
    - C is currently not allowed

    Parameters
    ----------
    axes :
        Axes to validate.

    Returns
    -------
    bool
        True if axes are valid, False otherwise.
    """
    _axes = axes.upper()

    # Minimum is 2 (XY) and maximum is 4 (TZYX)
    if len(_axes) < 2 or len(_axes) > 4:
        raise ValueError(
            f"Invalid axes {axes}. Must contain at least 2 and at most 4 axes."
        )

    # all characters must be in REF_AXES = 'STCZYX'
    if not all([s in AXES for s in _axes]):
        raise ValueError(f"Invalid axes {axes}. Must be a combination of {AXES}.")

    # check for repeating characters
    for i, s in enumerate(_axes):
        if i != _axes.rfind(s):
            raise ValueError(
                f"Invalid axes {axes}. Cannot contain duplicate axes (got multiple {axes[i]})."
            )

    # currently no implementation for C
    if "C" in _axes:
        raise NotImplementedError("Currently, C axis is not supported.")

    # prevent S and T axes together
    if "T" in _axes and "S" in _axes:
        raise NotImplementedError(
            f"Invalid axes {axes}. Cannot contain both S and T axes."
        )

    # prior: X and Y contiguous (#FancyComments)
    # right now the next check is invalidating this, but in the future, we might
    # allow random order of axes (or at least XY and YX)
    if not ("XY" in _axes) and not ("YX" in _axes):
        raise ValueError(f"Invalid axes {axes}. X and Y must be contiguous.")

    # check that the axes are in the right order
    for i, s in enumerate(_axes):
        if i < len(_axes) - 1:
            index_s = AXES.find(s)
            index_next = AXES.find(_axes[i + 1])

            if index_s > index_next:
                raise ValueError(
                    f"Invalid axes {axes}. Axes must be in the order {AXES}."
                )


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
            # TODO check min clip ?
            np.clip(n_patches[i] * patch_sizes[i] - arr.shape[i + 1], 0, None)
            / max(1, (n_patches[i] - 1))
        ).astype(int)
        for i in range(len(patch_sizes))
    ]
    return tuple(overlap)


def compute_overlap_auto(d, patch_size, min_overlap=30):
    for delta in range(min_overlap, patch_size - 1):
        if (d - patch_size) % (patch_size - delta) == 0:
            return patch_size - delta, delta

    return 1, patch_size - 1


def compute_overlap_predict(
    arr: np.ndarray, patch_size: Tuple[int], overlap: Tuple[int]
) -> Tuple[int]:
    steps, overlaps = [], []
    for i in range(len(patch_size)):
        step, overlap = compute_overlap_auto(
            arr.shape[i + 1], patch_size[i], overlap[i]
        )
        steps.append(step)
        overlaps.append(overlap)

    return [
        patch_size[i]
        - (arr.shape[i + 1] - total_patches[i] * (patch_size[i] - overlap[i]))
        for i in range(len(patch_size))
    ]


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
