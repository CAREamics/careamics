"""Test `CacheTiles` class."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.dataset import IterableTiledPredDataset
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    WriteTiles,
)

from .utils import create_tiles, patch_tile_cache


@pytest.fixture
def cache_tiles_strategy(write_func) -> WriteTiles:
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
    return WriteTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
        write_filenames=None,
        n_samples_per_file=None,
    )


# TODO: Move to test tile cache
def test_last_tiles(cache_tiles_strategy):
    """Test `CacheTiles.last_tile` property."""

    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    patch_tile_cache(cache_tiles_strategy.tile_cache, tiles, tile_infos)

    last_tiles = [False, False, False, False, False, False, False, False, True]
    cached_last_tiles = [
        tile_info.last_tile
        for tile_info in cache_tiles_strategy.tile_cache.tile_info_cache
    ]
    assert cached_last_tiles == last_tiles


def test_write_batch_no_last_tile(cache_tiles_strategy):
    """
    Test `CacheTiles.write_batch` when there is no last tile added to the cache.

    Expected behaviour is that the batch is added to the cache.
    """

    # all tiles of 1 samples with 9 tiles
    n_samples = 1
    tiles, tile_infos = create_tiles(n_samples=n_samples)

    # simulate adding a batch that will not contain the last tile
    n_tiles = 4
    batch_size = 2
    patch_tile_cache(
        cache_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
    )
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

    cache_tiles_strategy.set_file_data(
        write_filenames=["file_1.tiff"], n_samples_per_file=[n_samples]
    )
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
        np.array_equal(
            extended_tiles[i], cache_tiles_strategy.tile_cache.array_cache[i]
        )
        for i in range(n_tiles + batch_size)
    )
    assert extended_tile_infos == cache_tiles_strategy.tile_cache.tile_info_cache


def test_write_batch_last_tile(cache_tiles_strategy):
    """
    Test `CacheTiles.write_batch` when there is a last tile added to the cache.

    Expected behaviour is that the cache is cleared and the write func is called.
    """

    # all tiles of 2 samples with 9 tiles
    n_samples = 2
    tiles, tile_infos = create_tiles(n_samples=n_samples)

    # simulate adding a batch that will contain the last tile
    n_tiles = 8
    batch_size = 2
    patch_tile_cache(
        cache_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
    )
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

    file_names = [f"file_{i}" for i in range(n_samples)]
    n_samples_per_file = [1 for _ in range(n_samples)]

    # call write batch
    dirpath = Path("predictions")
    cache_tiles_strategy.set_file_data(
        write_filenames=file_names, n_samples_per_file=n_samples_per_file
    )

    # mocking stitch_prediction_single because to assert if WriteFunc was called
    with patch(
        "careamics.lightning.callbacks.prediction_writer_callback.write_strategy"
        + ".write_tiles.stitch_prediction_single",
    ) as mock_stitch_prediction_single:
        mock_prediction = np.arange(64).reshape(1, 1, 8, 8)
        mock_stitch_prediction_single.return_value = mock_prediction
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

    # assert write_func is called as expected
    write_func_call_args = cache_tiles_strategy.write_func.call_args.kwargs
    assert write_func_call_args["file_path"] == Path("predictions/file_0.ext")
    np.testing.assert_array_equal(write_func_call_args["img"], mock_prediction)

    # Tile of the next image (should remain in the cache)
    remaining_tile = tiles[9]
    remaining_tile_info = tile_infos[9]

    # assert cache cleared as expected
    assert len(cache_tiles_strategy.tile_cache.array_cache) == 1
    assert np.array_equal(
        remaining_tile, cache_tiles_strategy.tile_cache.array_cache[0]
    )
    assert remaining_tile_info == cache_tiles_strategy.tile_cache.tile_info_cache[0]


def test_write_batch_raises(cache_tiles_strategy: WriteTiles):
    """Test write batch raises a ValueError if the filenames have not been set."""
    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)

    # simulate adding a batch that will contain the last tile
    n_tiles = 8
    batch_size = 2
    patch_tile_cache(
        cache_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
    )
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

    with pytest.raises(ValueError):
        assert cache_tiles_strategy._write_filenames is None

        # call write batch
        dirpath = Path("predictions")
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


def test_reset(cache_tiles_strategy: WriteTiles):
    """Test CacheTiles.reset works as expected"""
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(cache_tiles_strategy.tile_cache, tiles[:-1], tile_infos[:-1])

    cache_tiles_strategy.set_file_data(write_filenames=["file"], n_samples_per_file=[1])
    cache_tiles_strategy.reset()

    assert cache_tiles_strategy._write_filenames is None
    assert cache_tiles_strategy._filename_iter is None
