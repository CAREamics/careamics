import numpy as np
from skimage.util import view_as_windows

from typing import Tuple


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


def compute_crop_and_stitch_coords_1d(
    axis_size: int, tile_size: int, overlap: int
) -> Tuple[Tuple[int]]:
    """Compute the coordinates for cropping image into tiles, cropping the overlap from predictions and stitching
    the tiles back together across one axis.

    Parameters
    ----------
    axis_size : int
        Length of the axis
    tile_size : int
        size of the tile for the given axis
    overlap : int
        size of the overlap for the given axis

    Returns
    -------
    Tuple[Tuple[int]]
        Tuple of all coordinates for given axis
    """
    axis_size = axis_size
    # Compute the step between tiles
    step = tile_size - overlap
    crop_coords = []
    stitch_coords = []
    overlap_crop_coords = []
    # Iterate over the axis with a certain step
    for i in range(0, axis_size + 1, step):
        # Check if the tile fits within the axis
        if i + tile_size <= axis_size:
            # Add the coordinates to crop one tile
            crop_coords.append((i, i + tile_size))
            # Add the pixel coordinates of the cropped tile in the image space
            stitch_coords.append(
                (i + overlap // 2 if i > 0 else 0, i + tile_size - overlap // 2)
            )
            # Add the coordinates to crop the overlap from the prediction
            overlap_crop_coords.append(
                (overlap // 2 if i > 0 else 0, tile_size - overlap // 2)
            )
        # If the tile does not fit within the axis, perform the abovementioned operations starting from the end of the axis
        else:
            crop_coords.append((axis_size - tile_size, axis_size))
            last_tile_end_coord = stitch_coords[-1][1]
            stitch_coords.append((last_tile_end_coord, axis_size))
            overlap_crop_coords.append(
                (tile_size - (axis_size - last_tile_end_coord), tile_size)
            )
            break
    return crop_coords, stitch_coords, overlap_crop_coords


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
    rng = np.random.default_rng()  # TODO not sure shuffling should be done here
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches
