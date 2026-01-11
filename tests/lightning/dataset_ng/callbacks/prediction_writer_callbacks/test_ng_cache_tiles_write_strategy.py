"""Test `CacheTiles` class."""

import random
from pathlib import Path

import numpy as np
import pytest

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import (
    TileSpecs,
    TilingStrategy,
)
from careamics.file_io.write import write_tiff
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    CachedTiles,
    create_write_file_path,
)


# TODO code is very similar to test_ng_stitching_prediction fixture, refactor
# Also this is going to be hard to maintain if anything changes in the way the dataset
# produces tiles
@pytest.fixture
def tiles(n_data, shape, axes) -> list[ImageRegionData]:
    # create data
    array = np.arange(n_data * np.prod(shape)).reshape((n_data, *shape))

    # create tiling strategy
    if "Z" in axes:
        tile_size = (8, 16, 16)
        overlaps = (2, 4, 4)
    else:
        tile_size = (16, 16)
        overlaps = (4, 4)

    if "S" in axes:
        if "C" in axes:
            shape_with_sc = shape
        else:
            shape_with_sc = (shape[0], 1, *shape[1:])
    else:
        if "C" in axes:
            shape_with_sc = (1, *shape)
        else:
            shape_with_sc = (1, 1, *shape)

    tiling_strategy = TilingStrategy(
        data_shapes=[shape_with_sc] * n_data, patch_size=tile_size, overlaps=overlaps
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    image_stacks = load_arrays(source=[array[i] for i in range(n_data)], axes=axes)
    patch_extractor = PatchExtractor(image_stacks)

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tile_spec: TileSpecs = tiling_strategy.get_patch_spec(i)
        tile = patch_extractor.extract_channel_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        tiles.append(
            ImageRegionData(
                data=tile,
                source=f"array_{i}.tif",
                dtype=str(tile.dtype),
                data_shape=shape_with_sc,
                axes=axes,
                region_spec=tile_spec,
                additional_metadata={},
            )
        )
    return tiles


@pytest.fixture
def cache_tiles_strategy() -> CachedTiles:
    """
    Initialized `CacheTiles` class.

    Returns
    -------
    CacheTiles
        Initialized `CacheTiles` class.
    """
    write_extension = ".tif"
    write_func_kwargs = {}
    write_func = write_tiff
    return CachedTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def update_cache(cache_strategy: CachedTiles, tiles: list[ImageRegionData]):
    """Helper function to patch the tile cache."""
    for tile in tiles:
        cache_strategy.tile_cache[tile.region_spec["data_idx"]].append(tile)


@pytest.mark.parametrize("n_data, shape, axes", [(1, (32, 32), "YX")])
def test_write_batch_incomplete(
    tiles: list[ImageRegionData], cache_tiles_strategy: CachedTiles
):
    """
    Test `CacheTiles.write_batch` when there is no last tile added to the cache.

    Expected behaviour is that the batch is added to the cache.
    """
    # simulate adding a batch that will not contain the last tile
    n_tiles = 4
    batch_size = 2
    update_cache(cache_tiles_strategy, tiles[:n_tiles])

    # create next batch
    next_batch = tiles[n_tiles : n_tiles + batch_size]

    cache_tiles_strategy.write_batch(
        dirpath="predictions",
        predictions=next_batch,
    )

    extended_tiles = tiles[: n_tiles + batch_size]
    assert len(extended_tiles) == len(cache_tiles_strategy.tile_cache[0])


# TODO mock as in previous versions of the test?
@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_write_batch_no_full_image(
    tiles: list[ImageRegionData], cache_tiles_strategy: CachedTiles, tmp_path: Path
):
    """
    Test `CacheTiles.write_batch` when the new batch contains the last tile.
    """
    n_tiles = len(tiles) // 2 - 1
    batch_size = 2
    update_cache(cache_tiles_strategy, tiles[:n_tiles])
    assert len(cache_tiles_strategy._get_full_images()) == 0

    # create next batch
    next_batch = tiles[n_tiles : n_tiles + batch_size]

    dirpath = tmp_path / "predictions"
    dirpath.mkdir(parents=True, exist_ok=True)

    cache_tiles_strategy.write_batch(
        dirpath=dirpath,
        predictions=next_batch,
    )

    # check existence of written file
    file_name = tiles[0].source
    expected_path = create_write_file_path(
        dirpath=dirpath,
        file_path=Path(file_name),
        write_extension=cache_tiles_strategy.write_extension,
    )
    assert expected_path.exists()

    # check remaining tiles in cache
    remaining_tiles = tiles[len(tiles) // 2 : n_tiles + batch_size]

    assert cache_tiles_strategy.tile_cache.keys() == {1}
    for i in range(len(remaining_tiles)):
        np.testing.assert_array_equal(
            cache_tiles_strategy.tile_cache[1][i].data, remaining_tiles[i].data
        )


@pytest.mark.parametrize("n_data, shape, axes", [(4, (28, 28), "YX")])
def test_get_full_images(
    tiles: list[ImageRegionData], cache_tiles_strategy: CachedTiles
):
    """Test `CacheTiles._get_full_images`."""
    # randomize tiles order and add a few tiles from other images
    random_tiles = [tiles[15], tiles[10]] + tiles[: len(tiles) // 2 + 1]
    # randomize
    random.shuffle(random_tiles)

    update_cache(cache_tiles_strategy, random_tiles)

    data_indices = cache_tiles_strategy._get_full_images()
    assert set(data_indices) == {0, 1}


@pytest.mark.parametrize("n_data, shape, axes", [(1, (28, 28), "YX")])
def test_get_full_images_too_many(
    tiles: list[ImageRegionData], cache_tiles_strategy: CachedTiles
):
    """Test `CacheTiles._get_full_images` raises error when too many tiles of a data_idx
    are cached."""
    # add all tiles plus one extra
    extra_tile = tiles[1]
    tiles_with_extra = tiles + [extra_tile]

    update_cache(cache_tiles_strategy, tiles_with_extra)

    with pytest.raises(ValueError):
        _ = cache_tiles_strategy._get_full_images()
