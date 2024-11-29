"""Test the utility `TileCache` class used by the WriteTile classes."""

import numpy as np
import pytest

from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    caches,
)

from .utils import create_tiles, patch_tile_cache


def test_add(): ...


def test_has_last_tile_true():
    """
    Test `TileCache.has_last_tile` returns true when there is a last tile.
    """

    tile_cache = caches.TileCache()
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)

    patch_tile_cache(tile_cache, tiles, tile_infos)
    assert tile_cache.has_last_tile()


def test_has_last_tile_false():
    """Test `TileCache.has_last_tile` returns false when there is not a last tile."""

    tile_cache = caches.TileCache()
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(tile_cache, tiles[:-1], tile_infos[:-1])

    assert not tile_cache.has_last_tile()


def test_pop_image_tiles():
    """
    Test `TileCache.has_last_tile` removes the tiles up until the first "last tile".
    """

    tile_cache = caches.TileCache()
    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)
    # include first tile from next sample
    patch_tile_cache(tile_cache, tiles[:10], tile_infos[:10])

    image_tiles, image_tile_infos = tile_cache.pop_image_tiles()

    # test popped tiles as expected
    assert len(image_tiles) == 9
    assert all(np.array_equal(image_tiles[i], tiles[i]) for i in range(9))
    assert image_tile_infos == tile_infos[:9]

    # test tiles remaining in cache as expected
    assert len(tile_cache.array_cache) == 1
    assert np.array_equal(tile_cache.array_cache[0], tiles[9])
    assert tile_cache.tile_info_cache[0] == tile_infos[9]


def test_pop_image_tiles_error():
    """Test `CacheTiles._last_tile_index` raises an error when there is no last tile."""

    tile_cache = caches.TileCache()
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(tile_cache, tiles[:-1], tile_infos[:-1])
    # should raise error if pop_image_tiles is called without last tile present
    with pytest.raises(ValueError):
        tile_cache.pop_image_tiles()


def test_reset(): ...
