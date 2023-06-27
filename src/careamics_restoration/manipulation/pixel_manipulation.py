from typing import Tuple

import numpy as np


def get_stratified_coords(mask_pixel_perc: float, shape: Tuple[int, ...]) -> np.ndarray:
    # TODO add description, add asserts, add typing, add line comments
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    mask_pixel_perc : float
        _description_
    shape : Tuple[int, ...]
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    rng = np.random.default_rng()

    # step = [(d / np.sqrt(d)).astype(np.int32) for d in shape]
    # Define the approximate distance between masked pixels
    box_size = np.round(np.sqrt(np.product(shape) / 100 * mask_pixel_perc)).astype(
        np.int32
    )

    # Define a grid of coordinates for each axis in the input patch and the step size
    pixel_coords, step = np.linspace(
        0, shape, box_size, dtype=np.int32, endpoint=False, retstep=True
    )

    # Create a meshgrid of coordinates for each axis in the input patch
    coordinate_grid_list = np.meshgrid(*pixel_coords.T.tolist())
    coordinate_grid = np.array(coordinate_grid_list).reshape(len(shape), -1).T

    # Add random jitter to the grid to account for cases where the step size is not an integer
    odd_jitter = np.where(np.floor(step) == step, 0, rng.integers(0, 2))

    # Define the random jitter to be added to the grid
    grid_random_increment = rng.integers(
        np.zeros_like(coordinate_grid),
        odd_jitter + np.floor(step) * np.ones_like(coordinate_grid).astype(np.int32),
        size=coordinate_grid.shape,
    )
    coordinate_grid += grid_random_increment
    return coordinate_grid


# TODO unused dims parameter
def apply_struct_n2v_mask(patch, coords, dims, mask):
    """
    each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value.
    """
    coords = np.array(coords).astype(np.int32)
    ndim = mask.ndim
    center = np.array(mask.shape) // 2
    ## leave the center value alone
    mask[tuple(center.T)] = 0j
    ## displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords[None]
    mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape) - 1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2
    return patch


def n2v_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    roi_size: int = 5,
    augmentations=None,
) -> Tuple[np.ndarray, ...]:
    """Manipulate pixel in a patch with N2V algorithm.

    Parameters
    ----------
    patch : np.ndarray
        image patch, 2D or 3D, shape (y, x) or (z, y, x)
    mask_pixel_percentage : floar
        Percentage of pixels to be masked, well kinda
    roi_size : int
        Size of ROI where to take replacement pixels from, by default 5
    augmentations : _type_, optional
        _description_, by default None

    Returns
    -------
    Tuple[np.ndarray]
        manipulated patch, original patch, mask
    """
    # TODO add better docstring, add augmentations, add support for 3D
    original_patch = patch.copy()
    # Get the coordinates of the pixels to be replaced
    roi_centers = get_stratified_coords(mask_pixel_percentage, patch.shape)
    rng = np.random.default_rng()
    # Generate coordinate grid for ROI
    roi_span_full = np.arange(-np.floor(roi_size / 2), np.ceil(roi_size / 2)).astype(
        np.int32
    )
    # Remove the center pixel from the grid
    roi_span_wo_center = roi_span_full[roi_span_full != 0]
    # Randomly select coordinates from the grid
    random_increment = rng.choice(roi_span_wo_center, size=roi_centers.shape)
    # Clip the coordinates to the patch size
    replacement_coords = np.clip(
        roi_centers + random_increment,
        0,
        [patch.shape[i] - 1 for i in range(len(patch.shape))],
    )
    # Get the replacement pixels from all rois
    replacement_pixels = patch[tuple(replacement_coords.T.tolist())]
    # Replace the original pixels with the replacement pixels
    patch[tuple(roi_centers.T.tolist())] = replacement_pixels
    mask = np.where(patch != original_patch, 1, 0)
    patch, original_patch, mask = (
        (patch, original_patch, mask)
        if augmentations is None
        else augmentations(patch, original_patch, mask)
    )
    return (
        np.expand_dims(patch, 0),
        np.expand_dims(original_patch, 0),
        np.expand_dims(mask, 0),
    )
