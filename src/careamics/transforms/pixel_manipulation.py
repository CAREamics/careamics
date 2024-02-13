"""
Pixel manipulation methods.

Pixel manipulation is used in N2V and similar algorithm to replace the value of
masked pixels.
"""
from typing import Optional, Tuple

import numpy as np


def _apply_struct_mask(
    patch: np.ndarray, coords: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Applies structN2V mask to patch.

    Each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value.

    Parameters
    ----------
    patch : np.ndarray
        Patch to be manipulated.
    coords : np.ndarray
        Coordinates of the pixels to be manipulated.
    mask : np.ndarray
        Mask to be applied.

    Returns
    -------
    TODO
    """
    mask = np.array(mask)
    ndim = mask.ndim
    center = np.array(mask.shape) // 2

    # leave the center value alone
    mask[tuple(center.T)] = 0

    # displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]

    # combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords.T[None]
    mix = mix.transpose([1, 0, 2]).reshape([ndim, -1]).T

    # stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim), max=np.array(patch.shape) - 1).astype(np.uint8)

    # replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.rand(mix.shape[0]) * 4 - 2

    return patch


def _odd_jitter_func(step: float, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly sample a jitter to be applied to the masking grid.

    This is done to account for cases where the step size is not an integer.

    Parameters
    ----------
    step : float
        Step size of the grid, output of np.linspace.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of random jitter to be added to the grid.
    """
    # Define the random jitter to be added to the grid
    odd_jitter = np.where(np.floor(step) == step, 0, rng.integers(0, 2))

    # Round the step size to the nearest integer depending on the jitter
    return np.floor(step) if odd_jitter == 0 else np.ceil(step)


def _get_stratified_coords(
    mask_pixel_perc: float,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Generate coordinates of the pixels to mask.

    Randomly selects the coordinates of the pixels to mask in a stratified way, i.e.
    the distance between masked pixels is approximately the same.

    Parameters
    ----------
    mask_pixel_perc : float
        Actual (quasi) percentage of masked pixels across the whole image. Used in
        calculating the distance between masked pixels across each axis.
    shape : Tuple[int, ...]
        Shape of the input patch.

    Returns
    -------
    np.ndarray
        Array of coordinates of the masked pixels.
    """
    rng = np.random.default_rng()

    # Define the approximate distance between masked pixels
    mask_pixel_distance = np.round((100 / mask_pixel_perc) ** (1 / len(shape))).astype(
        np.int32
    )

    # Define a grid of coordinates for each axis in the input patch and the step size
    pixel_coords = []
    for axis_size in shape:
        # make sure axis size is evenly divisible by box size
        num_pixels = int(np.ceil(axis_size / mask_pixel_distance))
        axis_pixel_coords, step = np.linspace(
            0, axis_size, num_pixels, dtype=np.int32, endpoint=False, retstep=True
        )
        # explain
        pixel_coords.append(axis_pixel_coords.T)

    # Create a meshgrid of coordinates for each axis in the input patch
    coordinate_grid_list = np.meshgrid(*pixel_coords)
    coordinate_grid = np.array(coordinate_grid_list).reshape(len(shape), -1).T

    grid_random_increment = rng.integers(
        _odd_jitter_func(float(step), rng)
        * np.ones_like(coordinate_grid).astype(np.int32)
        - 1,
        size=coordinate_grid.shape,
        endpoint=True,
    )
    coordinate_grid += grid_random_increment
    coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)
    return coordinate_grid


# TODO channels: masking the same pixel across channels?
def uniform_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    roi_size: int = 11,
    struct_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Manipulate pixel in a patch, i.e. replace the masked value.

    Parameters
    ----------
    patch : np.ndarray
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    roi_size : int
        Size of the ROI the new pixel value is sampled from, by default 11.
    augmentations : Callable, optional
        Augmentations to apply, by default None.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple containing the manipulated patch, the original patch and the mask.
    """
    # TODO this assumes patch has no channel dimension. Is this correct?
    patch = patch.squeeze()
    original_patch = patch.copy()

    # TODO: struct mask could be generated from roi center & removed from grid as well
    # Get the coordinates of the pixels to be replaced
    roi_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape)
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
    # TODO: shouldn't the maximum be roi_center+patch.shape/2? rather than path.shape
    replacement_coords: np.ndarray = np.clip(
        roi_centers + random_increment,
        0,
        [patch.shape[i] - 1 for i in range(len(patch.shape))],
    )
    # Get the replacement pixels from all rois
    replacement_pixels = patch[tuple(replacement_coords.T.tolist())]

    # Replace the original pixels with the replacement pixels
    patch[tuple(roi_centers.T.tolist())] = replacement_pixels

    # Create corresponding mask
    mask = np.where(patch != original_patch, 1, 0).astype(np.uint8)

    if struct_mask is not None:
        patch = _apply_struct_mask(patch, roi_centers, struct_mask)

    # Expand the dimensions of the arrays to return the channel dimension
    # TODO Should it be done here? done at all?
    return (
        np.expand_dims(patch, 0),
        np.expand_dims(
            original_patch, 0
        ),  # TODO is this necessary to return the original patch?
        np.expand_dims(mask, 0),
    )


# TODO: fix
# TODO: create tests
# TODO: find an optimized way in np without for loop
# TODO: add struct mask
def median_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    roi_size: int = 11,
    struct_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """Works on the assumption that it is 2D or 3D image."""
    # TODO this assumes patch has no channel dimension. Is this correct?
    patch = patch.squeeze()
    mask = np.zeros_like(patch)
    original_patch = patch.copy()

    # Get the coordinates of the pixels to be replaced
    roi_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape)

    for center in roi_centers:
        min_coord = [max(0, c - roi_size // 2) for c in center]
        max_coord = [min(s, c + roi_size // 2 + 1) for s, c in zip(patch.shape, center)]

        coords = [slice(min_coord[i], max_coord[i]) for i in range(patch.ndim)]

        # extract roi around center
        roi = patch[tuple(coords)]

        # replace center pixel by median
        patch[tuple(center)] = np.median(roi)
        mask[tuple(center)] = 1

    return (
        np.expand_dims(patch, 0),
        np.expand_dims(original_patch, 0),
        np.expand_dims(mask, 0),
    )
