"""A module for selecting data to be set aside for validation."""

import numpy as np

from .patching_strategies import (
    FixedPatchingStrategy,
    PatchSpecs,
    StratifiedPatchingStrategy,
)


def create_val_split(
    stratified_patching: StratifiedPatchingStrategy,
    n_val_patches: int,
    rng: np.random.Generator,
) -> tuple[StratifiedPatchingStrategy, FixedPatchingStrategy]:
    """
    Create patching strategies for training an validation.

    The patches from the training patching strategy will never overlap with the patches
    from the validation patching strategy.

    Parameters
    ----------
    stratified_patching : StratifiedPatchingStrategy
        The patching strategy to select and exclude validation patches from.
    n_val_patches: int,
        The number of validation patches.
    rng : int, optional
        An optional seed to ensure the reproducibility of the validation patch choice.
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

    # validation patches have to lie on this grid
    grid_coords = stratified_patching.get_included_grid_coords()
    keys = list(grid_coords.keys())
    val_patch_specs: list[PatchSpecs] = []

    # select validation patches
    selected_images = rng.choice(len(keys), n_val_patches)
    # the images and how many validation patches should be selected from each image
    selected_images, n_image_patches = np.unique(selected_images, return_counts=True)
    for idx, n_patches in zip(selected_images, n_image_patches, strict=True):

        data_idx, sample_idx = keys[idx]
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

        # collected the chosen validation patches to create the fixed patching strategy
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

    val_patching_strategy = FixedPatchingStrategy(val_patch_specs)
    return stratified_patching, val_patching_strategy
