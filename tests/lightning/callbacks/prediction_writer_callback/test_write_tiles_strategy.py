"""Test `CacheTiles` class."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.dataset import IterableTiledPredDataset
from careamics.file_io.write import write_tiff
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    WriteTiles,
)
from careamics.prediction_utils import stitch_prediction


@pytest.fixture
def write_tiles_strategy() -> WriteTiles:
    """
    Initialized `WriteTiles` class.

    Parameters
    ----------
    write_func : WriteFunc
        Write function. (Comes from fixture).

    Returns
    -------
    CacheTiles
        Initialized `CacheTiles` class.
    """
    write_extension = ".tiff"
    write_func_kwargs = {}
    return WriteTiles(
        write_func=write_tiff,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
        write_filenames=None,
        n_samples_per_file=None,
    )


def test_write_batch_no_last_tile(
    tmp_path,
    write_tiles_strategy: WriteTiles,
    create_tiles,
    patch_tile_cache,
):
    """
    Test behaviour when the last tile of a sample is not present.

    `WriteTiles.write_batch` should cache the tiles in the given batch.
    """
    # all tiles of 1 samples with 9 tiles
    n_samples = 1
    tiles, tile_infos = create_tiles(n_samples=n_samples)

    # simulate adding a batch that will not contain the last tile
    n_tiles = 4
    batch_size = 2
    patch_tile_cache(
        write_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
    )
    next_batch = (
        np.concatenate(tiles[n_tiles : n_tiles + batch_size]),
        tile_infos[n_tiles : n_tiles + batch_size],
    )

    # mock trainer and datasets (difficult to set up true classes)
    trainer: Trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    write_tiles_strategy.set_file_data(
        write_filenames=["file_1.tif"], n_samples_per_file=[n_samples]
    )
    write_tiles_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=next_batch,
        batch_indices=Mock(),
        batch=next_batch,  # does not contain the last tile
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath=tmp_path / "predictions",
    )

    extended_tiles = tiles[: n_tiles + batch_size]
    extended_tile_infos = tile_infos[: n_tiles + batch_size]

    # assert tiles and tile infos are caches
    for i in range(n_tiles + batch_size):
        np.testing.assert_array_equal(
            extended_tiles[i], write_tiles_strategy.tile_cache.array_cache[i]
        )
    assert extended_tile_infos == write_tiles_strategy.tile_cache.tile_info_cache
    assert len(write_tiles_strategy.sample_cache.sample_cache) == 0


def test_write_batch_has_last_tile_no_last_sample(
    tmp_path,
    write_tiles_strategy: WriteTiles,
    create_tiles,
    patch_tile_cache,
):
    """
    Test behaviour when the last tile of a sample is present, but not the last sample.

    `WriteTiles.write_batch` should cache the resulting stitched sample.
    """
    # all tiles of 2 samples with 9 tiles each
    n_samples = 2
    tiles, tile_infos = create_tiles(n_samples=n_samples)

    # simulate adding a batch that will not contain the last sample
    n_tiles = 8
    batch_size = 2
    patch_tile_cache(
        write_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
    )
    # (ext batch will include the last tile of the first sample
    next_batch = (
        np.concatenate(tiles[n_tiles : n_tiles + batch_size]),
        tile_infos[n_tiles : n_tiles + batch_size],
    )

    # mock trainer and datasets (difficult to set up true classes)
    trainer: Trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    write_tiles_strategy.set_file_data(
        write_filenames=["file_1.tif"], n_samples_per_file=[n_samples]
    )
    write_tiles_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=next_batch,
        batch_indices=Mock(),
        batch=next_batch,  # contains the last tile of the first sample
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath=tmp_path / "predictions",
    )

    stitched_sample1 = stitch_prediction(tiles[:9], tile_infos[:9])[0]

    # assert first tile from the second sample is in the tile cache
    np.testing.assert_array_equal(
        tiles[9], write_tiles_strategy.tile_cache.array_cache[0]
    )
    assert [tile_infos[9]] == write_tiles_strategy.tile_cache.tile_info_cache
    # assert the stitched sample is saved in the sample cache
    np.testing.assert_array_equal(
        stitched_sample1, write_tiles_strategy.sample_cache.sample_cache[0]
    )


def test_write_batch_has_last_sample(
    tmp_path,
    write_tiles_strategy: WriteTiles,
    create_tiles,
    patch_tile_cache,
):
    """
    Test behaviour when the last sample of a file is present.

    `WriteTiles.write_batch` should write the resulting set of samples to disk.
    """
    """
    Test behaviour when the last tile of a sample is present, but not the last sample.

    `WriteTiles.write_batch` should cache the resulting stitched sample.
    """

    # all tiles of 2 samples with 9 tiles each
    n_samples = 2
    tiles, tile_infos = create_tiles(n_samples=n_samples)

    stitched_samples = stitch_prediction(tiles, tile_infos)
    file_data = np.concatenate(stitched_samples)

    # simulate adding a batch that will not contain the last sample
    batch_size = 2
    n_tiles = 16
    write_tiles_strategy.set_file_data(
        write_filenames=["file_1.tif"], n_samples_per_file=[n_samples]
    )
    patch_tile_cache(
        write_tiles_strategy.tile_cache, tiles[9:n_tiles], tile_infos[9:n_tiles]
    )
    # also patch_sample
    write_tiles_strategy.sample_cache.sample_cache.append(stitched_samples[0])
    # (ext batch will include the last tile of the first sample
    next_batch = (
        np.concatenate(tiles[n_tiles : n_tiles + batch_size]),
        tile_infos[n_tiles : n_tiles + batch_size],
    )

    # mock trainer and datasets (difficult to set up true classes)
    trainer: Trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    # normally output directory creation is handled by PredictionWriterCallback
    (tmp_path / "predictions").mkdir()
    write_tiles_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=next_batch,
        batch_indices=Mock(),
        batch=next_batch,  # contains the last tile of the last sample
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath=tmp_path / "predictions",
    )

    # assert caches are now empty
    assert len(write_tiles_strategy.tile_cache.array_cache) == 0
    assert len(write_tiles_strategy.tile_cache.tile_info_cache) == 0
    assert len(write_tiles_strategy.sample_cache.sample_cache) == 0

    assert (tmp_path / "predictions" / "file_1.tiff").is_file()

    load_file_data = tifffile.imread(tmp_path / "predictions" / "file_1.tiff")
    np.testing.assert_array_equal(load_file_data, file_data)


def test_write_batch_raises(
    write_tiles_strategy: WriteTiles, create_tiles, patch_tile_cache
):
    """Test write batch raises a ValueError if the filenames have not been set."""
    # all tiles of 2 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=2)

    # simulate adding a batch that will contain the last tile
    n_tiles = 8
    batch_size = 2
    patch_tile_cache(
        write_tiles_strategy.tile_cache, tiles[:n_tiles], tile_infos[:n_tiles]
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
        assert write_tiles_strategy._write_filenames is None

        # call write batch
        dirpath = Path("predictions")
        write_tiles_strategy.write_batch(
            trainer=trainer,
            pl_module=Mock(spec=LightningModule),
            prediction=next_batch,
            batch_indices=Mock(),
            batch=next_batch,  # contains the last tile
            batch_idx=3,
            dataloader_idx=dataloader_idx,
            dirpath=dirpath,
        )


def test_reset(write_tiles_strategy: WriteTiles, create_tiles, patch_tile_cache):
    """Test CacheTiles.reset works as expected"""
    # all tiles of 1 samples with 9 tiles
    tiles, tile_infos = create_tiles(n_samples=1)
    # don't include last tile
    patch_tile_cache(write_tiles_strategy.tile_cache, tiles[:-1], tile_infos[:-1])

    write_tiles_strategy.set_file_data(write_filenames=["file"], n_samples_per_file=[1])
    write_tiles_strategy.reset()

    assert write_tiles_strategy._write_filenames is None
    assert write_tiles_strategy._filename_iter is None
