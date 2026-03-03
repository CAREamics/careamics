from collections.abc import Sequence

import pytest
from test_all_strategies import _create_fixed_patching_strategy


@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 16, 9)], (8, 8)],
        [[(2, 1, 64, 64), (1, 1, 43, 71), (3, 1, 14, 9)], (8, 5)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 16, 9, 12)], (8, 8, 8)],
        [[(2, 1, 64, 64, 64), (1, 1, 43, 71, 46), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_fixed_specs(data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]):
    """
    Assert that the same patch spec is always returned for each index.
    """
    patching_strategy = _create_fixed_patching_strategy(data_shapes, patch_size)
    patch_specs = [
        patching_strategy.get_patch_spec(i) for i in range(patching_strategy.n_patches)
    ]
    for idx in range(patching_strategy.n_patches):
        patch_spec = patching_strategy.get_patch_spec(idx)
        assert patch_spec == patch_specs[idx]
