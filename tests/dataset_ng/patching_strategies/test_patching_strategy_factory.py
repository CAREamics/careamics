from careamics.config.data.patching_strategies import (
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
)
from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    RandomPatchingStrategy,
    TilingStrategy,
    WholeSamplePatchingStrategy,
    create_patching_strategy,
)


def test_create_random_patching_strategy():
    data_shapes = [(1, 1, 32, 32, 32)]
    config = RandomPatchingConfig(name="random", patch_size=[16, 16, 16], seed=100)
    strategy = create_patching_strategy(data_shapes, config)
    assert isinstance(strategy, RandomPatchingStrategy)


def test_create_fixed_random_patching_strategy():
    data_shapes = [(1, 1, 32, 32, 32)]
    config = FixedRandomPatchingConfig(
        name="fixed_random", patch_size=[16, 16, 16], seed=100
    )
    strategy = create_patching_strategy(data_shapes, config)
    assert isinstance(strategy, FixedRandomPatchingStrategy)


def test_create_tiling_strategy():
    data_shapes = [(1, 1, 32, 32, 32)]
    config = TiledPatchingConfig(
        name="tiled", patch_size=[32, 32, 32], overlaps=[8, 8, 8]
    )
    strategy = create_patching_strategy(data_shapes, config)
    assert isinstance(strategy, TilingStrategy)


def test_create_whole_sample_patching_strategy():
    data_shapes = [(1, 1, 32, 32, 32)]
    config = WholePatchingConfig(name="whole")
    strategy = create_patching_strategy(data_shapes, config)
    assert isinstance(strategy, WholeSamplePatchingStrategy)
