"""Test `WriteImage` class."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.dataset import IterablePredDataset
from careamics.file_io import WriteFunc
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy import (
    WriteImage,
)


@pytest.fixture
def write_func():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteFunc)


@pytest.fixture
def write_image_strategy(write_func) -> WriteImage:
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
    return WriteImage(
        write_func=write_func,
        write_filenames=None,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def test_cache_tiles_init(write_func, write_image_strategy):
    """
    Test `WriteImage` initializes as expected.
    """
    assert write_image_strategy.write_func is write_func
    assert write_image_strategy.write_extension == ".ext"
    assert write_image_strategy.write_func_kwargs == {}


def test_write_batch(write_image_strategy, ordered_array):

    n_batches = 1

    prediction = ordered_array((n_batches, 1, 8, 8))

    batch = prediction
    batch_indices = np.arange(n_batches)
    batch_idx = 0

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)

    # mock trainer and datasets
    trainer = Mock(spec=Trainer)
    mock_dataset = Mock(spec=IterablePredDataset)
    dataloader_idx = 0
    mock_dl = Mock(spec=DataLoader)
    mock_dl.batch_size = 1
    trainer.predict_dataloaders = [mock_dl]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    # call write batch
    dirpath = Path("predictions")
    write_image_strategy.write_filenames = ["file"]
    write_image_strategy.write_batch(
        trainer=trainer,
        pl_module=Mock(spec=LightningModule),
        prediction=prediction,
        batch_indices=batch_indices,
        batch=batch,  # contains the last tile
        batch_idx=batch_idx,
        dataloader_idx=dataloader_idx,
        dirpath=dirpath,
    )

    # assert write_func is called as expected
    # cannot use `assert_called_once_with` because of numpy array
    write_image_strategy.write_func.assert_called_once()
    assert write_image_strategy.write_func.call_args.kwargs["file_path"] == Path(
        "predictions/file.ext"
    )
    np.testing.assert_array_equal(
        write_image_strategy.write_func.call_args.kwargs["img"], prediction
    )


def test_reset(write_image_strategy: WriteImage):
    """Test WriteImage.reset works as expected"""
    write_image_strategy.write_filenames = ["file"]
    write_image_strategy.current_file_index = 1
    write_image_strategy.reset()
    assert write_image_strategy.write_filenames is None
    assert write_image_strategy.current_file_index == 0
