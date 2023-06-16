import itertools
import numpy as np
from skimage.util import view_as_windows

from typing import Tuple


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
    rng = np.random.default_rng()  # TODO not sure shuffling should be done here
    patches = view_as_windows(arr, window_shape=window_shape, step=step).reshape(
        *output_shape
    )
    rng.shuffle(patches, axis=0)
    return patches
