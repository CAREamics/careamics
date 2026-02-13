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
    patch_size = stratified_patching.patch_size
    grid_coords = stratified_patching.get_included_grid_coords()
    keys = list(grid_coords.keys())
    val_patch_specs: list[PatchSpecs] = []

    selected_samples = rng.choice(len(keys), n_val_patches)
    selected_samples, n_sample_patches = np.unique(selected_samples, return_counts=True)
    for idx, n_patches in zip(selected_samples, n_sample_patches, strict=True):
        data_idx, sample_idx = keys[idx]
        coord_indices = rng.choice(
            len(grid_coords[(data_idx, sample_idx)]), n_patches, replace=False
        )
        coords: list[tuple[int, ...]] = [
            grid_coords[(data_idx, sample_idx)][coord_idx]
            for coord_idx in coord_indices
        ]
        stratified_patching.exclude_patches(data_idx, sample_idx, coords)

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
