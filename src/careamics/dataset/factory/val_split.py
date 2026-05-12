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
    if n_val_patches >= viable_patches_per_image.sum():
        raise ValueError(
            "The number of validation patches to be extracted from the training set is "
            f"too large for given training data size(got {n_val_patches} validation"
            f"patches but only {viable_patches_per_image.sum()} training patches "
            f"available with patch size {patch_size}). Make sure you have enough data "
            "to train with, decrease 'n_val_patches' or the patch size."
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
) -> list[tuple[int, int]]:
    if rng is None:
        rng = np.random.default_rng()
    coords = _create_validation_blocks(grid_shape, n_val_patches, rng)
    val_coords = _remove_excess_selected(grid_shape, coords, n_val_patches, rng)
    return [tuple(coord) for coord in val_coords]


def _create_validation_blocks(
    grid_shape: Sequence[int],
    n_val_patches: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    if rng is None:
        rng = np.random.default_rng()

    ndims = len(grid_shape)

    # calculate coordinates of validation patches in each dimension
    val_coords_1D = [None for _ in range(ndims)]
    for i in rng.permutation(np.arange(ndims)):
        dim_size: int = grid_shape[i]

        # calculate how many val coords in the current dimension
        remaining = np.ceil(
            n_val_patches
            / np.prod([len(coords) for coords in val_coords_1D if coords is not None])
        )
        ratios: list[float] = [  # ratios between dimensions
            grid_shape[i] / grid_shape[j]
            for j in range(ndims)
            if i != j and val_coords_1D[j] is None
        ]
        n_coords = int(
            np.ceil((remaining * np.prod(ratios)) ** (1 / (len(ratios) + 1)))
        )

        block_size, gap_size = _find_block_sequence_params(dim_size, n_coords)
        val_coords_1D[i] = _block_sequence(block_size, gap_size, dim_size)

        # add random offset
        if (edge_diff := dim_size - val_coords_1D[i][-1]) > 4:
            random_offset = rng.choice(np.concat([[0], np.arange(2, edge_diff - 2)]))
            val_coords_1D[i] += random_offset

        # randomly mirror sequence:
        if rng.integers(0, 1, endpoint=True):
            val_coords_1D[i] = dim_size - 1 - val_coords_1D[i]

    val_coords = np.stack(np.meshgrid(*val_coords_1D, indexing="ij"), axis=-1).reshape(
        -1, ndims
    )
    return val_coords


def _block_sequence(block_size: int, gap_size: int, max_value: int) -> NDArray[np.int_]:
    n_values = int(np.ceil(max_value / (block_size + gap_size))) * block_size
    sequence = np.array(
        # eqn that produces n consecutive values followed by a gap of size m
        # e.g. block_size = 2, gap_size = 3: [0, 1, 5, 6, 10, 11, ...]
        [int(i + gap_size * np.floor(i / block_size)) for i in range(n_values)]
    )
    sequence = sequence[sequence < max_value]
    return sequence


def _find_block_sequence_params(max_value: int, n_values: int) -> tuple[int, int]:
    # TODO: raise error for for too many n_values
    best = None
    for gap_size in range(2, max_value):
        n_full_periods = (max_value - n_values) // gap_size
        block_size = int(
            np.ceil((max_value - n_full_periods * gap_size) / (n_full_periods + 1))
        )
        period = block_size + gap_size
        remainder = max_value % period
        sequence_length = (max_value // period) * block_size + remainder

        # prevents gaps of less than 2 at the edges
        if remainder > block_size:
            continue
        if sequence_length < n_values:
            continue

        n_periods = max_value / period

        candidate = {
            "block_size": block_size,
            "gap_size": gap_size,
            "length_diff": sequence_length - n_values,
            "n_periods": n_periods,
        }
        if best is None:
            best = candidate

        if (
            candidate["length_diff"] <= best["length_diff"]
            # less than two periods pushes all the selection to the edges
            # we only want less than two blocks if the previous best had less than two
            and (
                best["n_periods"] <= 2
                or candidate["n_periods"] >= 2
                or best["gap_size"] == 2
            )
        ):
            best = candidate

        if (best["length_diff"] <= 1) and (
            best["gap_size"] >= 3 or candidate["n_periods"] <= 2
        ):
            break
    return best["block_size"], best["gap_size"]


def _remove_excess_selected(
    grid_shape: Sequence[int],
    coords: NDArray[np.int_],
    n_val_patches: int,
    rng: np.random.Generator,
):
    n_dims = len(grid_shape)
    padding = 2
    coord_map = np.zeros(grid_shape, dtype=bool)
    coord_map[*[coords[:, i] for i in range(n_dims)]] = True
    coord_map = np.pad(coord_map, padding, mode="constant", constant_values=True)
    selected: list[int] = rng.permutation(len(coords)).tolist()

    # randomly remove coords, but only those that do not create a gap of size 1
    while len(selected) > n_val_patches:
        removable: list[bool] = []

        for idx in rng.permutation(selected).copy():
            coord = coords[idx]
            is_removable = not _cannot_remove(coord, coord_map, padding)
            if is_removable:
                coord_map[
                    *[slice(coord[i], coord[i] + padding) for i in range(n_dims)]
                ] = False
                selected.remove(idx)
            removable.append(is_removable)
            if len(selected) <= n_val_patches:
                break

        if not any(removable):
            break

    return coords[selected]


def _cannot_remove(
    coord: np.ndarray[tuple[int], np.dtype[np.int_]],
    coord_map: NDArray[np.bool],
    padding: int = 2,
) -> bool:
    n_dims = coord_map.ndim

    # Remove coord
    coord_map[coord[0] + padding, coord[1] + padding] = False
    neigborhood = coord_map[
        *[slice(coord[i], coord[i] + 2 * padding + 1) for i in range(n_dims)]
    ]
    not_removable = False
    for axis in range(n_dims):
        axis_slice = tuple(Ellipsis if i == axis else padding for i in range(n_dims))
        not_removable = _contains_gap_size_1(neigborhood[axis_slice])
        if not_removable:
            break
    # replace
    coord_map[*[slice(coord[i], coord[i] + padding) for i in range(n_dims)]] = True
    return not_removable


def _contains_gap_size_1(arr: np.ndarray[tuple[int], np.dtype[np.bool_]]) -> bool:
    return np.any(arr[:-2] & ~arr[1:-1] & arr[2:]).item()


def _n_viable_val_patches(grid_shape: Sequence[int]):
    # return zero if negative
    return max(0, np.prod([gs - 2 for gs in grid_shape]))
