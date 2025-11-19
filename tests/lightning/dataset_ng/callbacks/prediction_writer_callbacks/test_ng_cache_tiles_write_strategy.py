"""Test `CacheTiles` class."""

from pathlib import Path

import numpy as np
import pytest

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patching_strategies import (
    TileSpecs,
    TilingStrategy,
)
from careamics.file_io.write import write_tiff
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    CacheTiles,
    create_write_file_path,
)

# TODO simulate real batches


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
        data_shapes=[shape_with_sc] * n_data, tile_size=tile_size, overlaps=overlaps
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    patch_extractor = create_array_extractor(
        source=[array[i] for i in range(n_data)], axes=axes
    )

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tile_spec: TileSpecs = tiling_strategy.get_patch_spec(i)
        tile = patch_extractor.extract_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        tiles.append(
            ImageRegionData(
                data=tile,
                source=f"array_{i}.tif",
                dtype=str(tile.dtype),
                data_shape=shape,
                axes=axes,
                region_spec=tile_spec,
            )
        )
    return tiles


@pytest.fixture
def cache_tiles_strategy() -> CacheTiles:
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
    return CacheTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def update_cache(cache_strategy: CacheTiles, tiles: list[ImageRegionData]):
    """Helper function to patch the tile cache."""
    cache_strategy.tile_cache.extend(tiles)
    cache_strategy.data_indices.extend([tile.region_spec["data_idx"] for tile in tiles])
    cache_strategy.last_tile.extend([tile.region_spec["last_tile"] for tile in tiles])


@pytest.mark.parametrize("n_data, shape, axes", [(1, (32, 32), "YX")])
def test_write_batch_no_last_tile(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
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
    extended_data_indices = [tile.region_spec["data_idx"] for tile in extended_tiles]
    extended_last_tiles = [tile.region_spec["last_tile"] for tile in extended_tiles]

    assert all(
        np.array_equal(extended_tiles[i].data, cache_tiles_strategy.tile_cache[i].data)
        for i in range(n_tiles + batch_size)
    )
    assert extended_data_indices == cache_tiles_strategy.data_indices
    assert extended_last_tiles == cache_tiles_strategy.last_tile


# TODO mock as in previous versions of the test?
@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_write_batch_with_last_tile(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles, tmp_path: Path
):
    """
    Test `CacheTiles.write_batch` when there is no last tile added to the cache.

    Expected behaviour is that the batch is added to the cache.
    """
    n_tiles = len(tiles) // 2 - 1
    batch_size = 2
    update_cache(cache_tiles_strategy, tiles[:n_tiles])
    assert not cache_tiles_strategy._has_last_tile()

    # create next batch
    next_batch = tiles[n_tiles : n_tiles + batch_size]

    cache_tiles_strategy.write_batch(
        dirpath=tmp_path,
        predictions=next_batch,
    )

    # check existence of written file
    file_name = tiles[0].source
    expected_path = create_write_file_path(
        dirpath=tmp_path,
        file_path=Path(file_name),
        write_extension=cache_tiles_strategy.write_extension,
    )
    assert expected_path.exists()

    # check remaining tiles in cache
    remaining_tiles = tiles[len(tiles) // 2 : n_tiles + batch_size]
    remaining_data_indices = [tile.region_spec["data_idx"] for tile in remaining_tiles]
    remaining_last_tiles = [tile.region_spec["last_tile"] for tile in remaining_tiles]

    for i, tile in enumerate(cache_tiles_strategy.tile_cache):
        np.testing.assert_array_equal(tile.data, remaining_tiles[i].data)

    assert remaining_data_indices == cache_tiles_strategy.data_indices
    assert remaining_last_tiles == cache_tiles_strategy.last_tile


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_last_tiles(tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles):
    """Test `CacheTiles._has_last_tile` property."""
    update_cache(cache_tiles_strategy, tiles[: len(tiles) // 2 + 1])

    assert cache_tiles_strategy._has_last_tile()


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_last_tiles_false(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
):
    """Test `CacheTiles._has_last_tile` property."""
    update_cache(cache_tiles_strategy, tiles[: len(tiles) // 2 - 1])

    assert not cache_tiles_strategy._has_last_tile()


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_find_last_tile_index(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
):
    """Test `CacheTiles.__find_last_tile_index`."""
    # There is two last tile, find the first one
    update_cache(cache_tiles_strategy, tiles)

    index = cache_tiles_strategy._find_last_tile_index()
    assert index == len(tiles) // 2 - 1


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_find_image_tile_indices(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
):
    """Test `CacheTiles._find_image_tile_indices`."""
    # There is two last tile, find the first one
    update_cache(cache_tiles_strategy, tiles)

    data_idx = tiles[0].region_spec["data_idx"]
    indices = cache_tiles_strategy._find_image_tile_indices(data_idx)
    assert indices == list(range(len(tiles) // 2))

    data_idx = tiles[-1].region_spec["data_idx"]
    indices = cache_tiles_strategy._find_image_tile_indices(data_idx)
    assert indices == list(range(len(tiles) // 2, len(tiles)))


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_pop_cache(tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles):
    """Test `CacheTiles._find_image_tile_indices`."""
    # There is two last tile, find the first one
    update_cache(cache_tiles_strategy, tiles)

    data_idx = tiles[0].region_spec["data_idx"]
    indices = cache_tiles_strategy._find_image_tile_indices(data_idx)
    popped_tiles = cache_tiles_strategy._pop_cache(indices)

    for i in range(len(popped_tiles)):
        np.testing.assert_array_equal(popped_tiles[i].data, tiles[i].data)

    assert len(cache_tiles_strategy.tile_cache) == len(tiles) // 2
    assert len(cache_tiles_strategy.data_indices) == len(tiles) // 2
    assert len(cache_tiles_strategy.last_tile) == len(tiles) // 2
    for i, tile in enumerate(cache_tiles_strategy.tile_cache):
        j = len(tiles) // 2 + i
        np.testing.assert_array_equal(tile.data, tiles[j].data)


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_extract_image_tiles_error(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
):
    """Test `CacheTiles._extract_image_tiles` raises an error when there is no last
    tile."""
    update_cache(cache_tiles_strategy, tiles[: len(tiles) // 2 - 1])

    with pytest.raises(ValueError):
        cache_tiles_strategy._extract_image_tiles()


@pytest.mark.parametrize("n_data, shape, axes", [(2, (28, 28), "YX")])
def test_extract_image_tiles(
    tiles: list[ImageRegionData], cache_tiles_strategy: CacheTiles
):
    """Test `CacheTiles._extract_image_tiles` returns the tiles of a single image."""
    # include first tile from next sample
    update_cache(cache_tiles_strategy, tiles[: len(tiles) // 2 + 1])

    image_tiles = cache_tiles_strategy._extract_image_tiles()

    assert len(image_tiles) == len(tiles) // 2

    for i in range(len(image_tiles)):
        np.testing.assert_array_equal(image_tiles[i].data, tiles[i].data)

    assert len(cache_tiles_strategy.tile_cache) == 1
    np.testing.assert_array_equal(
        cache_tiles_strategy.tile_cache[0].data, tiles[len(tiles) // 2].data
    )
