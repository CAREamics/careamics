"""A module for selecting data to be set aside for validation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from careamics.utils.logging import get_logger

from ..patching import (
    FixedPatching,
    PatchSpecs,
    StratifiedPatching,
)


def create_val_split(
    stratified_patching: StratifiedPatching,
    n_val_patches: int,
    rng: np.random.Generator,
) -> tuple[StratifiedPatching, FixedPatching]:
    """
    Create patching strategies for training and validation.

    Note that the provided `stratified_patching` instance will be modified. The patches
    from the training patching strategy will never overlap with the patches from the
    validation patching strategy.

    Parameters
    ----------
    stratified_patching : StratifiedPatchingStrategy
        The patching strategy to select and exclude validation patches from.
    n_val_patches : int
        The number of validation patches.
    rng : numpy.random.Generator
        Random number generator to ensure reproducibility of the validation patch
        choice.

    Returns
    -------
    training_patching_strategy : StratifiedPatchingStrategy
        The patching strategy to be used for training. Patches will be sampled in a
        stratified way, for each epoch. It excludes all the patches that should be used
        for validation.
    validation_patching_strategy : FixedPatchingStrategy
        The patching strategy to be used for validation. It will return the same patches
        every epoch.
    """
    logger = get_logger("Validation Splitting")
    patch_size = stratified_patching.patch_size

    # validation patches have to lie on this grid
    grid_coords = stratified_patching.get_included_grid_coords()
    # sample_ids are (data_idx, sample_idx)
    sample_ids = list(grid_coords.keys())

    # --- randomly decide how many validation patches will be selected from each sample
    viable_patches_per_image = np.array(
        [
            # Note: images with a (2 x 2) grid have no viable validation patches
            _n_viable_val_patches(
                stratified_patching.image_patching[data_idx][sample_idx].grid_shape
            )
            for data_idx, sample_idx in sample_ids
        ]
    )
    if n_val_patches > viable_patches_per_image.sum():
        viable_image_size = (ps * 2 for ps in patch_size)
        raise ValueError(
            "The number of validation patches to be extracted from the training set is "
            f"too large for given training data size (got {n_val_patches} validation "
            f"patches but only {viable_patches_per_image.sum()} viable validation "
            f"patches available with patch size {patch_size}). If your image size "
            f"is smaller than {viable_image_size}, no validation patches can be "
            "selected from it. Make sure you have enough data to train with, decrease "
            "'n_val_patches' or the patch size. Alternatively, provide a separate "
            "validation input."
        )
    val_patches_per_image = np.zeros_like(viable_patches_per_image)
    val_patch_specs: list[PatchSpecs] = []
    for _ in range(n_val_patches):
        probs = viable_patches_per_image / viable_patches_per_image.sum()
        idx = rng.choice(np.arange(len(viable_patches_per_image)), p=probs)
        # add to the selected and remove from the viable count
        val_patches_per_image[idx] += 1
        viable_patches_per_image[idx] -= 1

    # --- for each image select the validation patches
    n_selected = 0  # for logging (it may not be possible to select the exact n patches)
    for idx, n_patches in enumerate(val_patches_per_image):
        if n_patches == 0:
            continue

        data_idx, sample_idx = sample_ids[idx]

        # select validation patches
        grid_shape = stratified_patching.image_patching[data_idx][sample_idx].grid_shape
        coords = select_validation(grid_shape, n_patches, rng)
        # exclude the chosen validation patches from training
        stratified_patching.exclude_patches(data_idx, sample_idx, coords)
        n_selected += len(
            stratified_patching.image_patching[data_idx][sample_idx].excluded_patches
        )

        # collect the chosen validation patches to create the fixed patching strategy
        patch_specs: list[PatchSpecs] = [
            {
                "data_idx": data_idx,
                "sample_idx": sample_idx,
                "coords": tuple(np.array(grid_coord) * np.array(patch_size)),
                "patch_size": patch_size,
            }
            for grid_coord in coords
        ]
        val_patch_specs.extend(patch_specs)
    logger.info(
        f"Selected and split {n_selected} patches from the training data for "
        "validation."
    )

    val_patching_strategy = FixedPatching(val_patch_specs)
    return stratified_patching, val_patching_strategy


def select_validation(
    grid_shape: Sequence[int],
    n_val_patches: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[int]]:
    """
    Choose grid coordinates for validation patches.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Validation patches must lie on a grid. The grid shape is the floor of the image
        size divided by the patch size.
    n_val_patches : n_val_patches
        The number of validation patches to select.
    rng : numpy.random.Generator | None, default=None
        Random number generator for reproducibility.

    Returns
    -------
    list[Sequence[int]]
        A list of grid coordinates for validation patches.
    """
    if rng is None:
        rng = np.random.default_rng()
    # the validation coordinate blocks may contain more patches than requested
    coords = _create_validation_blocks(grid_shape, n_val_patches, rng)
    # randomly remove patches until we have the correct number of patches
    val_coords = _remove_excess_selected(grid_shape, coords, n_val_patches, rng)
    return [tuple(coord) for coord in val_coords]


def _create_validation_blocks(
    grid_shape: Sequence[int],
    n_val_patches: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Generate validation coordinates so that the coordinates are grouped in blocks.

    The blocks are at least to patch widths apart and evenly spaced.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Validation patches must lie on a grid. The grid shape is the floor of the image
        size divided by the patch size.
    n_val_patches : n_val_patches
        The number of validation patches to select.
    rng : numpy.random.Generator | None, default=None
        Random number generator for reproducibility.

    Returns
    -------
    NDArray[np.int_]
        Validation block coordinates. An N x D array of integers where N is the number
        of coordinates and D is the number of dimensions. N may be larger than the
        requested `n_val_patches`.
    """
    if rng is None:
        rng = np.random.default_rng()

    ndims = len(grid_shape)

    # calculate validation patch coordinates in each dimension
    val_coords_1D: list[NDArray[np.int_] | None] = [None for _ in range(ndims)]
    # order dimensions smallest to largest but random order if they are the same size
    for i in np.lexsort((rng.random(ndims), np.array(grid_shape))):
        dim_size: int = grid_shape[i]

        # --- calculate the number of coordinates in each dimension
        # try to have the patches distributed evenly in each dimension
        ratios: list[float] = [  # ratios between dimensions
            grid_shape[i] / grid_shape[j]
            for j in range(ndims)
            if i != j and val_coords_1D[j] is None
        ]
        # remaining val patches for the dimensions not calculated yet
        remaining = np.ceil(
            n_val_patches
            / np.prod(
                np.array(
                    [len(coords) for coords in val_coords_1D if coords is not None]
                )
            )
            # NOTE: prod([]) == 1
        )
        # distribute the remaining coords evenly between dimension
        n_coords = int(
            np.ceil((remaining * np.prod(ratios)) ** (1 / (len(ratios) + 1)))
        )

        # --- chose the validation coordinates
        # optimum block_size and gap_size
        block_size, gap_size = _find_block_sequence_params(dim_size, n_coords)
        val_coords_1D[i] = _block_sequence(block_size, gap_size, dim_size)

        # add random offset
        if (edge_diff := dim_size - val_coords_1D[i][-1]) > 4:
            random_offset = rng.choice(np.concat([[0], np.arange(2, edge_diff - 2)]))
            val_coords_1D[i] += random_offset

        # randomly mirror sequence (otherwise top left corner always selected)
        if rng.integers(0, 1, endpoint=True):
            val_coords_1D[i] = dim_size - 1 - val_coords_1D[i]

    val_coords: NDArray[np.int_] = np.stack(
        np.meshgrid(*val_coords_1D, indexing="ij"), axis=-1
    ).reshape(-1, ndims)
    return val_coords


def _block_sequence(block_size: int, gap_size: int, max_value: int) -> NDArray[np.int_]:
    """Create a block sequence.

    This is a sequence of `block_size` consecutive integers followed by a `gap_size`
    until the next set of integer values, until the `max_value`.

    Parameters
    ----------
    block_size : int
        The length of consecutive sets of integers in the sequence.
    gap_size : int
        The length of the gap between consecutive sets of integers.
    max_value : int
        The maximum of all the values of the sequence.

    Returns
    -------
    NDArray[np.int_]
        The 1D sequence of integers.
    """
    n_values = int(np.ceil(max_value / (block_size + gap_size))) * block_size
    sequence = np.array(
        # eqn that produces n consecutive values followed by a gap of size m
        # e.g. block_size = 2, gap_size = 3: [0, 1, 5, 6, 10, 11, ...]
        [int(i + gap_size * np.floor(i / block_size)) for i in range(n_values)]
    )
    sequence = sequence[sequence < max_value]
    return sequence


def _find_block_sequence_params(max_value: int, n_values: int) -> tuple[int, int]:
    """
    Find a `block_size` and `gap_size` for a block sequence.

    Given `max_value` and `n_values` we can find a `block_size` and a `gap_size` such
    that:
    - The `gap_size` is greater than (inclusive) 2 and greater than (inclusive) 3 if
    possible.
    - The gap between an element and the edge must not be 1. (It Can be 0 or greater
    than 1).
    - The sequence length is greater than (inclusive) but as close to `n_values` as
      possible.

    Parameters
    ----------
    max_value : int
        The maximum of all the values of the sequence.
    n_values : int
        The desired number of elements in the sequence.

    Returns
    -------
    tuple[int, int]
        The (`block_size`, `gap_size`) to parametrize the block sequence.
    """
    # iterate through the block sizes until the best candidate is found
    # we want that:
    # - the gap between values an the edge must not be 1.
    # - the length of the generated sequence closely matches the desired n_values
    # - ideally that the gap_size >= 3 (but must be >= 2)
    # - ideally that the selected coords represent both the edge and centre of the image
    if n_values == 0:
        raise ValueError("Cannot choose block parameters for `n_values=0`.")
    # specific case for small grid size:
    # (current algorithm would instead put a validation patch in each corner)
    if max_value == 4 and n_values == 2:
        return 2, 2

    best = None
    for block_size in range(1, max_value + 1):
        total_periods = np.ceil(n_values / block_size)

        # calculate the gap_size from the block size
        gap_size = max(
            2, np.ceil((max_value - total_periods * block_size) / total_periods)
        )

        period = block_size + gap_size
        remainder = max_value % period
        sequence_length = (max_value // period) * block_size + min(
            block_size, remainder
        )
        if sequence_length < n_values:
            continue
        # if the block is not touching the edge there must be a gap of at least 2
        if block_size < remainder and remainder - block_size <= 2:
            continue

        candidate = {
            "block_size": block_size,
            "gap_size": gap_size,
            "length_diff": sequence_length - n_values,
        }

        if best is None:
            # the first candidate (smallest block size that matches the constraints)
            best = candidate

        # only change from the first candidate if certain constraints are met
        # prefer a gap size > 2
        improved_gap_size = best["gap_size"] == 2 and candidate["gap_size"] > 2
        if (
            candidate["length_diff"] <= best["length_diff"]
            and improved_gap_size
            and max_value / period >= 2
        ):
            best = candidate

        # NOTE: length_diff <= 1 is best for not easily divisible numbers
        #   - otherwise values will likely get pushed to the edges
        if best["length_diff"] <= 1 and best["gap_size"] >= 3:
            break
    assert best is not None  # best is always assigned the first candidate
    return int(best["block_size"]), int(best["gap_size"])


def _remove_excess_selected(
    grid_shape: Sequence[int],
    coords: NDArray[np.int_],
    n_val_patches: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Remove excess coordinates without creating an invalid selection.

    There must not be a gap of size 1 between two validation patches.

    Parameters
    ----------
    grid_shape : Sequence[int]
        The shape of the validation coordinate grid.
    coords : NDArray[np.int_]
        The coordinates of validation patches.
    n_val_patches : int
        The number of desired validation patches. Shape (N x D) where D is dimensions.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    NDArray[np.int_]
        The reduced validation coordinates.
    """
    n_dims = len(grid_shape)
    padding = 2
    coord_map = np.zeros(grid_shape, dtype=bool)
    coord_map[*[coords[:, i] for i in range(n_dims)]] = True
    # The coordinate map needs to be padded because we also cannot have a gap of 1
    # between a coordinate and the edge.
    coord_map = np.pad(coord_map, padding, mode="constant", constant_values=True)
    selected: list[int] = rng.permutation(len(coords)).tolist()

    # randomly remove coords, but only those that do not create a gap of size 1
    while len(selected) > n_val_patches:
        removable: list[bool] = []

        for idx in rng.permutation(selected).copy():
            coord = coords[idx]
            is_removable = _coord_is_removable(coord, coord_map, padding)
            if is_removable:
                coord_map[*[c + padding for c in coord]] = False
                selected.remove(idx)
            removable.append(is_removable)
            if len(selected) <= n_val_patches:
                break

        if not any(removable):
            break

    return coords[selected]


def _coord_is_removable(
    coord: np.ndarray[tuple[int], np.dtype[np.int_]],
    coord_map: NDArray[np.bool],
    padding: int = 2,
) -> bool:
    """Test if a coordinate is removable.

    Parameters
    ----------
    coord : np.ndarray[tuple[int], np.dtype[np.int_]]
        The candidate coordinate.
    coord_map : NDArray[np.bool]
        Coordinate map (padded with True values). Shows the location of the selected
        coordinates on a grid. The coordinate map needs to be padded because we also
        cannot have a gap of 1 between a coordinate and the edge.
    padding : int, default=True
        The padding size of the coordinate map.

    Returns
    -------
    bool
        True if the coordinate can be removed.
    """
    # testing removability by removing a coordinate and checking if it creates a gap

    n_dims = coord_map.ndim
    # Remove coord
    coord_map[*[c + padding for c in coord]] = False
    neigborhood = coord_map[
        *[slice(coord[i], coord[i] + 2 * padding + 1) for i in range(n_dims)]
    ]
    removable = True
    # has to not create a gap in any dimension
    for axis in range(n_dims):
        axis_slice = tuple(Ellipsis if i == axis else padding for i in range(n_dims))
        if _contains_gap_size_1(neigborhood[axis_slice]):
            removable = False
        if not removable:
            break
    # replace
    coord_map[*[c + padding for c in coord]] = True
    return removable


def _contains_gap_size_1(arr: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> bool:
    """Return whether a sequence of bools contains a gap of size 1.

    A gap of size 1 would be for example: `[True, True, False, True]`.

    Parameters
    ----------
    arr : np.ndarray
        1D array of booleans.

    Returns
    -------
    bool
        Whether a gap of size 1 is contained within the sequence.
    """
    return np.any(arr[:-2] & ~arr[1:-1] & arr[2:]).item()


def _n_viable_val_patches(grid_shape: Sequence[int]) -> int:
    """The maximum number of validation patches allowed for `grid_shape`.

    Parameters
    ----------
    grid_shape : Sequence[int]
        Validation patches must lie on a grid. The grid shape is the floor of the image
        size divided by the patch size.

    Returns
    -------
    int
        The maximum allowed number of validation patches.
    """
    # no viable validation patches for (2 x 2)
    if all(gs <= 2 for gs in grid_shape):
        return 0

    # for a grid size <= 2, the number of viable patches the full shape
    # ^ this is mostly to allow for a small Z dimension
    return np.prod([gs - 2 if gs > 2 else gs for gs in grid_shape]).item()
