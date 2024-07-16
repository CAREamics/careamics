"""Test `CacheTiles` class."""

from pathlib import Path
from unittest.mock import DEFAULT, Mock, patch

import numpy as np
import pytest
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.tile_information import TileInformation
from careamics.dataset import IterableTiledPredDataset
from careamics.dataset.tiling import extract_tiles
from careamics.file_io import WriteFunc
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    CacheTiles,
)


def create_tiles(n_samples: int) -> tuple[list[NDArray], list[TileInformation]]:
    """
    Create a set of tiles from `n_samples`.

    To create the tiles the following parameters, `tile_size=(4, 4)` and
    `overlaps=(2, 2)`, on an input array with shape (`n_samples`, 1, 8, 8); this
    results in 9 tiles per sample.

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate the tiles from.

    Returns
    -------
    tuple of (list of NDArray), list of TileInformation))
        Tuple where first element is the list of tiles and second element is a list
        of corresponding tile information.
    """

    input_shape = (n_samples, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = np.arange(np.prod(input_shape)).reshape(input_shape)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    return tiles, tile_infos


def patch_tile_cache(
    strategy: CacheTiles, tiles: list[NDArray], tile_infos: list[TileInformation]
) -> None:
    """
    Patch simulated tile cache into `strategy`.

    Parameters
    ----------
    strategy : CacheTiles
        Write strategy `CacheTiles`.
    tiles : list of NDArray
        Tiles to patch into `strategy.tile_cache`.
    tile_infos : list of TileInformation
        Corresponding tile information to patch into `strategy.tile_info_cache`.
    """
    strategy.tile_cache = tiles
    strategy.tile_info_cache = tile_infos


@pytest.fixture
def write_func():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteFunc)


@pytest.fixture
def cache_tiles_strategy(write_func) -> CacheTiles:
    """
    Initialized `CacheTiles` class.

    Parameters
    ----------
    write_func : WriteFunc
        Write function. (Comes from fixture).

    Returns
    -------
    CacheTiles
        Initialized `CacheTiles` class.
    """
    write_extension = ".ext"
    write_func_kwargs = {}
    return CacheTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def test_cache_tiles_init(write_func, cache_tiles_strategy):
    """
    Test `CacheTiles` initializes as expected.
    """
    assert cache_tiles_strategy.write_func is write_func
    assert cache_tiles_strategy.write_extension == ".ext"
    assert cache_tiles_strategy.write_func_kwargs == {}
    assert cache_tiles_strategy.tile_cache == []
    assert cache_tiles_strategy.tile_info_cache == []


def test_last_tiles(cache_tiles_strategy):
    """Test `CacheTiles.last_tile` property."""

    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    patch_tile_cache(cache_tiles_strategy, tiles, tile_infos)

    last_tile = [False, False, False, False, False, False, False, False, True]
    assert cache_tiles_strategy.last_tiles == last_tile


def test_write_batch_no_last_tile(cache_tiles_strategy):
    """
    Test `CacheTiles.write_batch` when there is no last tile added to the cache.

    Expected behaviour is that the batch is added to the cache.
    """

    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)

    # simulate adding a batch that will not contain the last tile
    n_tiles = 4
    batch_size = 2
    patch_tile_cache(cache_tiles_strategy, tiles[:n_tiles], tile_infos[:n_tiles])
    next_batch = (
        np.concatenate(tiles[n_tiles : n_tiles + batch_size]),
        tile_infos[n_tiles : n_tiles + batch_size],
    )

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    cache_tiles_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=next_batch,
        batch_indices=Mock(),
        batch=next_batch,  # does not contain the last tile
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath="predictions",
    )

    extended_tiles = tiles[: n_tiles + batch_size]
    extended_tile_infos = tile_infos[: n_tiles + batch_size]

    assert all(
        np.array_equal(extended_tiles[i], cache_tiles_strategy.tile_cache[i])
        for i in range(n_tiles + batch_size)
    )
    assert extended_tile_infos == cache_tiles_strategy.tile_info_cache


def test_write_batch_last_tile(cache_tiles_strategy):
    """
    Test `CacheTiles.write_batch` when there is a last tile added to the cache.

    Expected behaviour is that the cache is cleared and the write func is called.
    """

    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)

    # simulate adding a batch that will contain the last tile
    n_tiles = 8
    batch_size = 2
    patch_tile_cache(cache_tiles_strategy, tiles[:n_tiles], tile_infos[:n_tiles])
    next_batch = (
        np.concatenate(tiles[n_tiles : n_tiles + batch_size]),
        tile_infos[n_tiles : n_tiles + batch_size],
    )

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    # These functions have their own unit tests,
    #   so they do not need to be tested again here.
    # This is a unit test to isolate functionality of `write_batch.`
    with patch.multiple(
        "careamics.lightning.callbacks.prediction_writer_callback.write_strategy",
        stitch_prediction_single=DEFAULT,
        get_sample_file_path=DEFAULT,
        create_write_file_path=DEFAULT,
    ) as values:

        # mocked functions
        mock_stitch_prediction_single = values["stitch_prediction_single"]
        mock_get_sample_file_path = values["get_sample_file_path"]
        mock_create_write_file_path = values["create_write_file_path"]

        prediction_image = [Mock()]
        in_file_path = Path("in_dir/file_path.ext")
        out_file_path = Path("out_dir/file_path.in_ext")
        mock_stitch_prediction_single.return_value = prediction_image
        mock_get_sample_file_path.return_value = in_file_path
        mock_create_write_file_path.return_value = out_file_path

        # call write batch
        dirpath = "predictions"
        cache_tiles_strategy.write_batch(
            trainer=trainer,
            pl_module=Mock(spec=LightningModule),
            prediction=next_batch,
            batch_indices=Mock(),
            batch=next_batch,  # contains the last tile
            batch_idx=3,
            dataloader_idx=dataloader_idx,
            dirpath=dirpath,
        )

        # assert create_write_file_path is called as expected (TODO: necessary ?)
        mock_create_write_file_path.assert_called_once_with(
            dirpath=dirpath,
            file_path=in_file_path,
            write_extension=cache_tiles_strategy.write_extension,
        )
        # assert write_func is called as expected
        cache_tiles_strategy.write_func.assert_called_once_with(
            file_path=out_file_path, img=prediction_image[0], **{}
        )

    # Tile of the next image (should remain in the cache)
    remaining_tile = tiles[9]
    remaining_tile_info = tile_infos[9]

    # assert cache cleared as expected
    assert len(cache_tiles_strategy.tile_cache) == 1
    assert np.array_equal(remaining_tile, cache_tiles_strategy.tile_cache[0])
    assert remaining_tile_info == cache_tiles_strategy.tile_info_cache[0]


def test_have_last_tile_true(cache_tiles_strategy):
    """Test `CacheTiles._have_last_tile` returns true when there is a last tile."""

    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    patch_tile_cache(cache_tiles_strategy, tiles, tile_infos)

    assert cache_tiles_strategy._has_last_tile()


def test_have_last_tile_false(cache_tiles_strategy):
    """Test `CacheTiles._have_last_tile` returns false when there is not a last tile."""

    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(cache_tiles_strategy, tiles[:-1], tile_infos[:-1])

    assert not cache_tiles_strategy._has_last_tile()


def test_clear_cache(cache_tiles_strategy):
    """
    Test `CacheTiles._clear_cache` removes the tiles up until the first "last tile".
    """

    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)
    # include first tile from next sample
    patch_tile_cache(cache_tiles_strategy, tiles[:10], tile_infos[:10])

    cache_tiles_strategy._clear_cache()

    assert len(cache_tiles_strategy.tile_cache) == 1
    assert np.array_equal(cache_tiles_strategy.tile_cache[0], tiles[9])
    assert cache_tiles_strategy.tile_info_cache[0] == tile_infos[9]


def test_last_tile_index(cache_tiles_strategy):
    """Test `CacheTiles._last_tile_index` returns the index of the last tile."""
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    patch_tile_cache(cache_tiles_strategy, tiles, tile_infos)

    assert cache_tiles_strategy._last_tile_index() == 8


def test_last_tile_index_error(cache_tiles_strategy):
    """Test `CacheTiles._last_tile_index` raises an error when there is no last tile."""
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(cache_tiles_strategy, tiles[:-1], tile_infos[:-1])

    with pytest.raises(ValueError):
        cache_tiles_strategy._last_tile_index()


def test_get_image_tiles(cache_tiles_strategy):
    """Test `CacheTiles._get_image_tiles` returns the tiles of a single image."""
    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)
    # include first tile from next sample
    patch_tile_cache(cache_tiles_strategy, tiles[:10], tile_infos[:10])

    image_tiles, image_tile_infos = cache_tiles_strategy._get_image_tiles()

    assert len(image_tiles) == 9
    assert all(np.array_equal(image_tiles[i], tiles[i]) for i in range(9))
    assert image_tile_infos == tile_infos[:9]
