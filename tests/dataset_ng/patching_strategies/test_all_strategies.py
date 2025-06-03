from collections.abc import Callable, Sequence

import numpy as np
import pytest

from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    RandomPatchingStrategy,
    SequentialPatchingStrategy,
    TilingStrategy,
    WholeSamplePatchingStrategy,
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


def _create_sequential_patching_strategy(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> SequentialPatchingStrategy:
    overlap = tuple(2 for _ in patch_size)
    return SequentialPatchingStrategy(data_shapes, patch_size, overlap)


def _create_tiling_strategy(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> TilingStrategy:
    if len(patch_size) == 2:
        overlaps = (2, 2)
    elif len(patch_size) == 3:
        overlaps = (2, 2, 2)
    else:
        raise ValueError
    return TilingStrategy(
        data_shapes=data_shapes, tile_size=patch_size, overlaps=overlaps
    )


def _create_whole_sample_patching_strategy(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> WholeSamplePatchingStrategy:
    # patch_size unused
    return WholeSamplePatchingStrategy(data_shapes=data_shapes)


PatchingStrategyConstr = Callable[
    [Sequence[Sequence[int]], Sequence[int]], PatchingStrategy
]

# !!! if new strategies are added they should be tested here !!!
PATCHING_STRATEGY_CONSTR: tuple[PatchingStrategyConstr, ...] = (
    _create_random_patching_strategy,
    _create_fixed_random_patching_strategy,
    _create_sequential_patching_strategy,
    _create_tiling_strategy,
    _create_whole_sample_patching_strategy,
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
def test_all_get_patch_spec(
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
        assert (coords + patch_size_ <= spatial_shape).all()


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
def test_patches_cover_50percent(
    strategy_constr: PatchingStrategyConstr,
    data_shapes: Sequence[Sequence[int]],
    patch_size: Sequence[int],
):
    # Testing more than 50% because some patching strategies are random (but seeded)
    """Test that more than 50% of the data is covered by the sampled pathches"""
    patching_strategy = strategy_constr(data_shapes, patch_size)

    # track where patches have been sampled from
    tracking_arrays = [np.zeros(data_shape, dtype=bool) for data_shape in data_shapes]

    patch_specs = [
        patching_strategy.get_patch_spec(i) for i in range(patching_strategy.n_patches)
    ]
    for patch_spec in patch_specs:
        tracking_array = tracking_arrays[patch_spec["data_idx"]]
        spatial_slice = tuple(
            slice(c, c + ps)
            for c, ps in zip(
                patch_spec["coords"], patch_spec["patch_size"], strict=False
            )
        )
        # set to true where the patches would be sampled from
        tracking_array[(patch_spec["sample_idx"], slice(None), *spatial_slice)] = True

    total_covered = 0
    total_size = 0
    for tracking_array in tracking_arrays:
        # if the patch specs covered all the image all the values should be true
        total_covered += np.count_nonzero(tracking_array)
        total_size += tracking_array.size
    assert total_covered / total_size > 0.5
