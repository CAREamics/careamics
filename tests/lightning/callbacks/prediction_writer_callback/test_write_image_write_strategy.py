"""Test `WriteImage` class."""

from pathlib import Path
from unittest.mock import DEFAULT, Mock, patch

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

    # These functions have their own unit tests,
    #   so they do not need to be tested again here.
    # This is a unit test to isolate functionality of `write_batch.`
    with patch.multiple(
        "careamics.lightning.callbacks.prediction_writer_callback.write_strategy",
        get_sample_file_path=DEFAULT,
        create_write_file_path=DEFAULT,
    ) as values:

        # mocked functions
        mock_get_sample_file_path = values["get_sample_file_path"]
        mock_create_write_file_path = values["create_write_file_path"]

        # assign mock functions return value
        in_file_path = Path("in_dir/file_path.ext")
        out_file_path = Path("out_dir/file_path.out_ext")
        mock_get_sample_file_path.return_value = in_file_path
        mock_create_write_file_path.return_value = out_file_path

        # call write batch
        dirpath = "predictions"
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

        # assert create_write_file_path is called as expected (TODO: necessary ?)
        mock_create_write_file_path.assert_called_with(
            dirpath=dirpath,
            file_path=in_file_path,
            write_extension=write_image_strategy.write_extension,
        )
        # assert write_func is called as expectedÏ
        # cannot use `assert_called_once_with` because of numpy array
        write_image_strategy.write_func.assert_called_once()
        assert (
            write_image_strategy.write_func.call_args.kwargs["file_path"]
            == out_file_path
        )
        assert np.array_equal(
            write_image_strategy.write_func.call_args.kwargs["img"], prediction[0]
        )
