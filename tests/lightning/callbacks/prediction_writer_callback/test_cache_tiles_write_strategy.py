from pathlib import Path
from typing import Any, Optional, Protocol, Sequence
from unittest.mock import Mock, patch

import pytest
import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.tile_information import TileInformation
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.file_io import WriteFunc
from careamics.prediction_utils import stitch_prediction_single
from careamics.dataset.tiling import extract_tiles
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    CacheTiles
)

# TODO: add docs

@pytest.fixture
def write_func():
    return Mock(spec=WriteFunc)


@pytest.fixture
def cache_tiles_strategy(write_func):
    write_extension = ".ext"
    write_func_kwargs = {}
    return CacheTiles(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def test_cache_tiles_init(write_func, cache_tiles_strategy):

    assert cache_tiles_strategy.write_func is write_func
    assert cache_tiles_strategy.write_extension == ".ext"
    assert cache_tiles_strategy.write_func_kwargs == {}
    assert cache_tiles_strategy.tile_cache == []
    assert cache_tiles_strategy.tile_info_cache == []


def test_last_tiles(cache_tiles_strategy, ordered_array):

    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles
    cache_tiles_strategy.tile_info_cache = tile_infos

    last_tile = [False, False, False, False, False, False, False, False, True]
    assert all([cache_tiles_strategy.last_tiles[i] == last_tile[i] for i in range(9)])


def test_write_batch_no_last_tile(cache_tiles_strategy, ordered_array):

    # TODO: tidy up

    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    idx = 4
    batch_size = 2
    tile_cache = tiles[:idx]
    tile_info_cache = tile_infos[:idx]
    next_batch = (
        np.concatenate(tiles[idx : idx + batch_size]),
        tile_infos[idx : idx + batch_size],
    )

    cache_tiles_strategy.tile_cache = tile_cache
    cache_tiles_strategy.tile_info_cache = tile_info_cache

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
        batch=next_batch,
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath="predictions",
    )

    extended_tiles = tiles[: idx + batch_size]
    extended_tile_infos = tile_infos[: idx + batch_size]

    assert all(
        [
            np.array_equal(extended_tiles[i], cache_tiles_strategy.tile_cache[i])
            for i in range(idx + batch_size)
        ]
    )
    assert all(
        [
            extended_tile_infos[i] == cache_tiles_strategy.tile_info_cache[i]
            for i in range(idx + batch_size)
        ]
    )


def test_write_batch_last_tile(cache_tiles_strategy, ordered_array):

    # TODO: tidy up

    input_shape = (2, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    idx = 8
    batch_size = 2
    tile_cache = tiles[:idx]
    tile_info_cache = tile_infos[:idx]
    next_batch = (
        np.concatenate(tiles[idx : idx + batch_size]),
        tile_infos[idx : idx + batch_size],
    )

    cache_tiles_strategy.tile_cache = tile_cache
    cache_tiles_strategy.tile_info_cache = tile_info_cache

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterableTiledPredDataset)
    src_files = [Path("in_dir/test_file_0.old_ext"), Path("in_dir/test_file_1.old_ext")]
    mock_dataset.data_files = src_files
    dataloader_idx = 0
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset
    dirpath = "predictions"
    write_extension = cache_tiles_strategy.write_extension
    out_files = [
        (Path(dirpath) / fn.stem).with_suffix(write_extension) for fn in src_files
    ]

    cache_tiles_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=next_batch,
        batch_indices=Mock(),
        batch=next_batch,
        batch_idx=3,
        dataloader_idx=dataloader_idx,
        dirpath=dirpath,
    )

    with (
        patch(
            "careamics.lightning.callbacks.prediction_writer_callback"
            ".write_strategy.stitch_prediction_single"
        ) as mock_stitch_prediction_single
    ):
        prediction_image = [Mock()]
        mock_stitch_prediction_single.return_value = prediction_image

        cache_tiles_strategy.write_batch(
            trainer=trainer,
            pl_module=Mock(spec=LightningModule),
            prediction=next_batch,
            batch_indices=Mock(),
            batch=next_batch,
            batch_idx=3,
            dataloader_idx=dataloader_idx,
            dirpath="predictions",
        )

        cache_tiles_strategy.write_func.assert_called_once_with(
            file_path=out_files[0], img=prediction_image[0], **{}
        )

    remaining_tile = tiles[9]
    remaining_tile_info = tile_infos[9]

    assert len(cache_tiles_strategy.tile_cache) == 1
    assert np.array_equal(remaining_tile, cache_tiles_strategy.tile_cache[0])
    assert remaining_tile_info == cache_tiles_strategy.tile_info_cache[0]


def test_have_last_tile_true(cache_tiles_strategy, ordered_array):
    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles
    cache_tiles_strategy.tile_info_cache = tile_infos

    assert cache_tiles_strategy._have_last_tile()

def test_have_last_tile_false(cache_tiles_strategy, ordered_array):
    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles[:-1]
    cache_tiles_strategy.tile_info_cache = tile_infos[:-1]

    assert not cache_tiles_strategy._have_last_tile()

def test_clear_cache(cache_tiles_strategy, ordered_array):

    input_shape = (2, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles[:10]
    cache_tiles_strategy.tile_info_cache = tile_infos[:10]

    cache_tiles_strategy._clear_cache()
    
    assert len(cache_tiles_strategy.tile_cache) == 1
    assert np.array_equal(cache_tiles_strategy.tile_cache[0], tiles[9])
    assert cache_tiles_strategy.tile_info_cache[0] == tile_infos[9]

def test_last_tile_index(cache_tiles_strategy, ordered_array):
    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles
    cache_tiles_strategy.tile_info_cache = tile_infos

    assert cache_tiles_strategy._last_tile_index() == 8

def test_last_tile_index_error(cache_tiles_strategy, ordered_array):
    input_shape = (1, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles[:-1]
    cache_tiles_strategy.tile_info_cache = tile_infos[:-1]

    with pytest.raises(ValueError):
        cache_tiles_strategy._last_tile_index()

def test_get_image_tiles(cache_tiles_strategy, ordered_array):
    input_shape = (2, 1, 8, 8)
    tile_size = (4, 4)
    tile_overlap = (2, 2)

    arr = ordered_array(input_shape, dtype=int)

    all_tiles = list(extract_tiles(arr, tile_size, tile_overlap))
    tiles = [output[0] for output in all_tiles]
    tile_infos = [output[1] for output in all_tiles]

    cache_tiles_strategy.tile_cache = tiles[:10]
    cache_tiles_strategy.tile_info_cache = tile_infos[:10]

    image_tiles, image_tile_infos = cache_tiles_strategy._get_image_tiles()

    assert len(image_tiles) == 9
    assert all([
        np.array_equal(image_tiles[i], tiles[i]) for i in range(9)
    ])
    assert all([
        image_tile_infos[i] == tile_infos[i] for i in range(9)
    ])