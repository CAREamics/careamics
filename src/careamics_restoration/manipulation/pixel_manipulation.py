from typing import Callable, Optional, Tuple

import numpy as np


def odd_jitter_func(step: float, rng: np.random.Generator) -> np.ndarray:
    """Adds random jitter to the grid.

    This is done to account for cases where the step size is not an integer.

    Parameters
    ----------
    step : float
        Step size of the grid, output of np.linspace
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        array of random jitter to be added to the grid
    """
    # Define the random jitter to be added to the grid
    odd_jitter = np.where(np.floor(step) == step, 0, rng.integers(0, 2))
    # Round the step size to the nearest integer depending on the jitter
    return np.floor(step) if odd_jitter == 0 else np.ceil(step)


def get_stratified_coords(
    mask_pixel_perc: float, shape: Tuple[int, ...], seed: int = 42
) -> np.ndarray:
    """Get coordinates of the pixels to mask.

    Randomly selects the coordinates of the pixels to mask in a stratified way, i.e.
    the distance between masked pixels is approximately the same

    Parameters
    ----------
    mask_pixel_perc : float
        Actual (quasi) percentage of masked pixels across the whole image. Used in
        calculating the distance between masked pixels across each axis
    shape : Tuple[int, ...]
        Shape of the input patch

    Returns
    -------
    np.ndarray
        array of coordinates of the masked pixels
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
        odd_jitter_func(float(step), rng)
        * np.ones_like(coordinate_grid).astype(np.int32)
        - 1,
        size=coordinate_grid.shape,
        endpoint=True,
    )
    coordinate_grid += grid_random_increment
    coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)
    return coordinate_grid


def default_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    roi_size: int = 5,
    augmentations: Optional[Callable] = None,
    seed: int = 42,  # TODO seed is not used
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
