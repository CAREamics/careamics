"""A module for selecting data to be set aside for validation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

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
    patch_size = stratified_patching.patch_size
    if n_val_patches >= stratified_patching.n_patches:
        raise ValueError(
            "The number of validation patches to be extracted from the training set is "
            f"too large for given training data size(got {n_val_patches} validation"
            f"patches but only {stratified_patching.n_patches} training patches "
            f"available with patch size {patch_size}). Make sure you have enough data "
            "to train with, decrease 'n_val_patches' or the patch size."
        )

    # validation patches have to lie on this grid
    grid_coords = stratified_patching.get_included_grid_coords()
    # sample_ids are (data_idx, sample_idx)
    sample_ids = list(grid_coords.keys())

    # select validation patches
    n_patches_per_image = np.array(
        [
            stratified_patching.image_patching[data_idx][sample_idx].n_patches
            for data_idx, sample_idx in sample_ids
        ]
    )
    n_selected_image_patches = np.zeros_like(n_patches_per_image)
    val_patch_specs: list[PatchSpecs] = []
    for _ in range(n_val_patches):
        probs = n_patches_per_image / n_patches_per_image.sum()
        idx = rng.choice(np.arange(len(n_patches_per_image)), p=probs)
        n_selected_image_patches[idx] += 1
        n_patches_per_image[idx] -= 1

    for idx, n_patches in enumerate(n_selected_image_patches):
        data_idx, sample_idx = sample_ids[idx]
        # randomly choose the validation patches in the image
        coord_indices = rng.choice(
            len(grid_coords[(data_idx, sample_idx)]), n_patches, replace=False
        )
        coords: list[tuple[int, ...]] = [
            grid_coords[(data_idx, sample_idx)][coord_idx]
            for coord_idx in coord_indices
        ]
        # exclude the chosen validation patches from training
        stratified_patching.exclude_patches(data_idx, sample_idx, coords)

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

    val_patching_strategy = FixedPatching(val_patch_specs)
    return stratified_patching, val_patching_strategy


def select_validation(
    grid_shape: Sequence[int],
    n_val_patches: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    if rng is None:
        rng = np.random.default_rng()

    n_dims = len(grid_shape)

    coords = _create_validation_blocks(grid_shape, n_val_patches, rng)

    padding = 2
    coord_map = np.zeros(grid_shape, dtype=bool)
    coord_map[coords[:, 0], coords[:, 1]] = True
    coord_map = np.pad(coord_map, padding, mode="constant", constant_values=True)
    selected: list[int] = np.random.permutation(len(coords)).tolist()

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
        # At least 1 of the dimensions cannot have more than half be validation patches
        if all(coords is None for coords in val_coords_1D):
            n_coords = min(n_coords, dim_size // 2)

        # calculate the validation blocks
        gap_size = 3 if dim_size > 8 else 2  # initial gap size
        if n_coords < dim_size:
            block_size = int(np.ceil(gap_size * n_coords / (dim_size - n_coords)))
        else:
            # cannot have more coordinates than the size of the dimension
            # TODO: raise warning?
            block_size = dim_size

        # recalculate gap size, might want to increase it
        gap_size = int(np.ceil(block_size * (dim_size - n_coords) / n_coords))
        gap_size = max(gap_size, 2)  # not sure if this is needed

        val_coords_1D[i] = _val_block_sequence_1D(
            dim_size, block_size, gap_size, rng=rng
        )

    val_coords = np.stack(np.meshgrid(*val_coords_1D, indexing="ij"), axis=-1).reshape(
        -1, ndims
    )
    return val_coords


def _val_block_sequence_1D(
    dim_size: int,
    block_size: int,
    gap_size: int,
    edge_dist: int = 2,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    if rng is None:
        rng = np.random.default_rng()

    n_patches = int(np.ceil(dim_size / (block_size + gap_size))) * block_size
    sequence = np.array(
        # eqn that produces n consecutive values followed by a gap of size m
        # e.g. n = 2, m = 3: [0, 1, 5, 6, 10, 11, ...]
        [int(i + gap_size * np.floor(i / block_size)) for i in range(n_patches)]
    )
    sequence = sequence[sequence < dim_size]

    # if the final the final element is not touching the edge
    if sequence[-1] != dim_size - 1:
        # remove values which are closer than `edge_dist` to the edge
        sequence = sequence[dim_size - sequence > edge_dist]

    # add a random offset (stops single values from being at the edge)
    edge_dist_true = dim_size - sequence[-1]
    if edge_dist_true // 2 > 2:
        centre_offset = rng.choice(
            np.concat([[0], np.arange(2, edge_dist_true // 2 + 1)])
        )
        sequence += centre_offset

    # random chance to mirror sequence
    if rng.integers(0, 2):
        sequence = dim_size - 1 - sequence
    return sequence
