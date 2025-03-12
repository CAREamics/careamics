from collections.abc import Sequence
from typing import Callable

import numpy as np
import pytest

from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    RandomPatchingStrategy,
)


def _create_random_patching_strategy(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> RandomPatchingStrategy:
    seed = 42
    return RandomPatchingStrategy(
        data_shapes=data_shapes, patch_size=patch_size, seed=seed
    )


def _create_fixed_random_patching_strategy(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> FixedRandomPatchingStrategy:
    seed = 42
    return FixedRandomPatchingStrategy(
        data_shapes=data_shapes, patch_size=patch_size, seed=seed
    )


PatchingStrategyConstr = Callable[
    [Sequence[Sequence[int]], Sequence[int]], PatchingStrategy
]

# !!! if new strategies are added they should be tested here !!!
PATCHING_STRATEGY_CONSTR: tuple[PatchingStrategyConstr, ...] = (
    _create_random_patching_strategy,
    _create_fixed_random_patching_strategy,
)


@pytest.mark.parametrize("strategy_constr", PATCHING_STRATEGY_CONSTR)
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 8)],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 5)],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 8, 8)],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_get_all_patch_specs(
    strategy_constr: PatchingStrategyConstr,
    data_shapes: Sequence[Sequence[int]],
    patch_size: Sequence[int],
):
    strategy = strategy_constr(data_shapes, patch_size)
    n_patches = strategy.n_patches
    for i in range(n_patches):
        patch_spec = strategy.get_patch_spec(i)

        # validate patch is within spatial bounds
        coords = np.array(patch_spec["coords"])
        patch_size_ = np.array(patch_spec["patch_size"])
        data_index = patch_spec["data_idx"]
        data_shape = data_shapes[data_index]
        spatial_shape = data_shape[2:]

        assert (0 <= coords).all()
        assert (coords + patch_size_ < spatial_shape).all()
