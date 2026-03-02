import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from careamics.dataset_ng.patching_strategies import StratifiedPatchingStrategy


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 16, 9)], (8, 8)],
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 14, 9)], (8, 5)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 16, 9, 12)], (8, 8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_excluded_patches_never_selected(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
):
    """Ensure excluded patches are never sampled from."""
    rng = np.random.default_rng(42)
    patching_strategy = StratifiedPatchingStrategy(data_shapes, patch_size, 42)

    excluded_masks = [np.zeros(shape, dtype=bool) for shape in data_shapes]
    tracking_arrays = [np.zeros(shape, dtype=int) for shape in data_shapes]

    # exclude patches, choose randomly
    for data_idx, data_shape in enumerate(data_shapes):
        for sample_idx in range(data_shape[0]):
            grid_shape = patching_strategy.image_patching[data_idx][
                sample_idx
            ].grid_shape
            # exclude 1/8 th of patches
            n = int(np.ceil(np.prod(grid_shape) / 8))
            indices = rng.choice(np.prod(grid_shape), n)
            grid_coords = list(itertools.product(*[range(s) for s in grid_shape]))
            selected = [grid_coords[idx] for idx in indices]

            patching_strategy.exclude_patches(data_idx, sample_idx, selected)
            for grid_coord in selected:

                exluded_slice = [
                    slice(c * ps, (c + 1) * ps)
                    for c, ps in zip(grid_coord, patch_size, strict=True)
                ]
                excluded_masks[data_idx][sample_idx, ..., *exluded_slice] = True

    epochs = 5
    for _ in range(epochs):
        for i in range(patching_strategy.n_patches):
            patch_spec = patching_strategy.get_patch_spec(i)
            data_idx = patch_spec["data_idx"]
            sample_idx = patch_spec["sample_idx"]
            coord = patch_spec["coords"]
            patch_slice = [
                slice(c, c + ps) for c, ps in zip(coord, patch_size, strict=True)
            ]
            tracking_arrays[data_idx][sample_idx, ..., *patch_slice] += 1

    for excluded_mask, tracking_array in zip(
        excluded_masks, tracking_arrays, strict=True
    ):
        assert (tracking_array[excluded_mask] == 0).all()
        # for good measure make sure the rest of the array isn't zeros
        assert (tracking_array[~excluded_mask] != 0).any()
