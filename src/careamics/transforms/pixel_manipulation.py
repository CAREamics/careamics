"""
Pixel manipulation methods.

Pixel manipulation is used in N2V and similar algorithm to replace the value of
masked pixels.
"""
from typing import List, Optional, Tuple

import numpy as np


def _apply_struct_mask(patch: np.ndarray, coords: np.ndarray, mask_params: List[int]):
    """Applies structN2V mask to patch.

    Each point in coords corresponds to the center of the mask.
    then for point in the mask with value=1 we assign a random value.

    Parameters
    ----------
    patch : np.ndarray
        Patch to be manipulated.
    coords : np.ndarray
        Coordinates of the ROI(subpatch) centers.
    mask_params : list
        Axis and span across center for the structN2V mask.
    """
    struct_axis, struct_span = mask_params
    # Create a mask array
    mask = np.expand_dims(np.ones(struct_span), axis=list(range(len(patch.shape) - 1)))
    # Move the struct axis to the first position for indexing
    mask = np.moveaxis(mask, 0, struct_axis)
    center = np.array(mask.shape) // 2
    # Mark the center
    mask[tuple(center.T)] = 0
    # displacements from center
    dx = np.indices(mask.shape)[:, mask == 1] - center[:, None]
    # combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = dx.T[..., None] + coords.T[None]
    mix = mix.transpose([1, 0, 2]).reshape([mask.ndim, -1]).T
    # stay within patch boundary
    # TODO this will fail if center is on the edge!
    mix = mix.clip(min=np.zeros(mask.ndim), max=np.array(patch.shape) - 1).astype(
        np.uint8
    )
    # replace neighbouring pixels with random values from flat dist
    patch[tuple(mix.T)] = np.random.uniform(patch.min(), patch.max(), size=mix.shape[0])
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

    # Define the approximate distance between masked pixels. Subtracts 1 form the shape
    # to account for the channel dimension
    mask_pixel_distance = np.round(
        (100 / mask_pixel_perc) ** (1 / (len(shape) - 1))
    ).astype(np.int32)

    # Define a grid of coordinates for each axis in the input patch and the step size
    pixel_coords = []
    steps = []
    for axis_size in shape:
        # make sure axis size is evenly divisible by box size
        num_pixels = int(np.ceil(axis_size / mask_pixel_distance))
        axis_pixel_coords, step = np.linspace(
            0, axis_size, num_pixels, dtype=np.int32, endpoint=False, retstep=True
        )
        # explain
        pixel_coords.append(axis_pixel_coords.T)
        steps.append(step)

    # Create a meshgrid of coordinates for each axis in the input patch
    coordinate_grid_list = np.meshgrid(*pixel_coords)
    coordinate_grid = np.array(coordinate_grid_list).reshape(len(shape), -1).T

    grid_random_increment = rng.integers(
        _odd_jitter_func(float(max(steps)), rng)
        * np.ones_like(coordinate_grid).astype(np.int32)
        - 1,
        size=coordinate_grid.shape,
        endpoint=True,
    )
    coordinate_grid += grid_random_increment
    coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)
    return coordinate_grid


def _create_subpatch_center_mask(
    subpatch: np.ndarray, center_coords: np.ndarray
) -> np.ndarray:
    """Create a mask with the center of the subpatch masked.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.

    Returns
    -------
    np.ndarray
        Mask with the center of the subpatch masked.
    """
    mask = np.ones(subpatch.shape)
    mask[tuple(center_coords.T)] = 0
    return np.ma.make_mask(mask)


def _create_subpatch_struct_mask(
    subpatch: np.ndarray, center_coords: np.ndarray, struct_mask_params: List[int]
) -> np.ndarray:
    """Create a structN2V mask for the subpatch.

    Parameters
    ----------
    subpatch : np.ndarray
        Subpatch to be manipulated.
    center_coords : np.ndarray
        Coordinates of the original center before possible crop.
    struct_mask_params : list
        Axis and span across center for the structN2V mask.

    Returns
    -------
    np.ndarray
        StructN2V mask for the subpatch.
    """
    # Create a mask with the center of the subpatch masked
    mask_placeholder = np.ones(subpatch.shape)
    struct_axis, struct_span = struct_mask_params
    # reshape to move the struct axis to the first position
    mask_reshaped = np.moveaxis(mask_placeholder, struct_axis, 0)
    # create the mask index for the struct axis
    mask_index = slice(
        max(0, center_coords.take(struct_axis) - (struct_span - 1) // 2),
        min(
            1 + center_coords.take(struct_axis) + (struct_span - 1) // 2,
            subpatch.shape[struct_axis],
        ),
    )
    mask_reshaped[struct_axis][mask_index] = 0
    # reshape back to the original shape
    mask = np.moveaxis(mask_reshaped, 0, struct_axis)
    return np.ma.make_mask(mask)


def uniform_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    remove_center: bool = True,
    struct_mask_params: Optional[List[int]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Manipulate pixel in a patch, i.e. replace the masked value.

    Parameters
    ----------
    patch : np.ndarray
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    remove_center : bool
        Whether to remove the center pixel from the subpatch, by default False. See
        uniform with/without central pixel in the documentation. #TODO add link
    struct_mask_params : optional
        Axis and span across center for the structN2V mask.

    Returns
    -------
    Tuple[np.ndarray]
        Tuple containing the manipulated patch, the original patch and the mask.
    """
    # Get the coordinates of the pixels to be replaced
    transformed_patch = patch.copy()

    subpatch_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape)
    rng = np.random.default_rng()

    # Generate coordinate grid for subpatch
    roi_span_full = np.arange(
        -np.floor(subpatch_size / 2), np.ceil(subpatch_size / 2)
    ).astype(np.int32)

    # Remove the center pixel from the grid if needed
    roi_span = roi_span_full[roi_span_full != 0] if remove_center else roi_span_full

    # Randomly select coordinates from the grid
    random_increment = rng.choice(roi_span, size=subpatch_centers.shape)

    # Clip the coordinates to the patch size
    replacement_coords = np.clip(
        subpatch_centers + random_increment,
        0,
        [patch.shape[i] - 1 for i in range(len(patch.shape))],
    )
    # Get the replacement pixels from all subpatchs
    replacement_pixels = patch[tuple(replacement_coords.T.tolist())]
    struct_mask = None
    # Replace the original pixels with the replacement pixels
    transformed_patch[tuple(subpatch_centers.T.tolist())] = replacement_pixels
    mask = np.where(transformed_patch != patch, 1, 0).astype(np.uint8)

    if struct_mask_params is not None:
        transformed_patch = _apply_struct_mask(
            transformed_patch, subpatch_centers, struct_mask
        )

    return (
        transformed_patch,
        mask,
    )


def median_manipulate(
    patch: np.ndarray,
    mask_pixel_percentage: float,
    subpatch_size: int = 11,
    struct_mask_params: Optional[List[int]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Manipulate pixel in a patch, i.e. replace the masked value.

    Parameters
    ----------
    patch : np.ndarray
        Image patch, 2D or 3D, shape (y, x) or (z, y, x).
    mask_pixel_percentage : floar
        Approximate percentage of pixels to be masked.
    subpatch_size : int
        Size of the subpatch the new pixel value is sampled from, by default 11.
    struct_mask_params: optional,
        Axis and span across center for the structN2V mask.

    Returns
    -------
    Tuple[np.ndarray]
           Tuple containing the manipulated patch, the original patch and the mask.
    """
    transformed_patch = patch.copy()
    # Get the coordinates of the pixels to be replaced
    subpatch_centers = _get_stratified_coords(mask_pixel_percentage, patch.shape)

    # Generate coordinate grid for subpatch
    roi_span = np.array(
        [-np.floor(subpatch_size / 2), np.ceil(subpatch_size / 2)]
    ).astype(np.int32)

    subpatch_crops_span_full = subpatch_centers[np.newaxis, ...].T + roi_span

    # Dimensions n dims, n centers, (min, max)
    subpatch_crops_span_clipped = np.clip(
        subpatch_crops_span_full,
        a_min=np.zeros_like(patch.shape)[:, np.newaxis, np.newaxis],
        a_max=np.array(patch.shape)[:, np.newaxis, np.newaxis] - 1,
    )

    for idx in range(subpatch_crops_span_clipped.shape[1]):
        subpatch_coords = subpatch_crops_span_clipped[:, idx, ...]
        idxs = [
            slice(x[0], x[1]) if x[1] - x[0] > 0 else slice(0, 1)
            for x in subpatch_coords
        ]
        subpatch = patch[tuple(idxs)]
        if struct_mask_params is None:
            subpatch_mask = _create_subpatch_center_mask(
                subpatch, subpatch_centers[idx]
            )
        else:
            subpatch_mask = _create_subpatch_struct_mask(
                subpatch, subpatch_centers[idx], struct_mask_params
            )
        transformed_patch[tuple(subpatch_centers[idx].tolist())] = np.median(
            subpatch[subpatch_mask]
        )

    mask = np.where(transformed_patch != patch, 1, 0).astype(np.uint8)

    if struct_mask_params is not None:
        transformed_patch = _apply_struct_mask(
            transformed_patch, subpatch_centers, struct_mask_params
        )

    return (
        transformed_patch,
        mask,
    )
