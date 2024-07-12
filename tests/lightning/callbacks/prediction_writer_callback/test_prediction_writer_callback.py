"""Test PredictionWriterCallback class."""

import os
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import pytest
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.support import SupportedData
from careamics.dataset import IterablePredDataset
from careamics.lightning.callbacks import PredictionWriterCallback
from careamics.lightning.callbacks.prediction_writer_callback import WriteStrategy


@pytest.fixture
def write_strategy():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteStrategy)


@pytest.fixture
def dirpath(tmp_path: Path):
    """Directory path."""
    return tmp_path / "predictions"


@pytest.fixture
def prediction_writer_callback(
    write_strategy: WriteStrategy, dirpath: Union[Path, str]
):
    """Initialized `PredictionWriterCallback`."""
    return PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath)


# TODO: smoke test with tiff (& example custom save func?)


def test_initialization(prediction_writer_callback, write_strategy, dirpath):
    """Test `PredictionWriterCallback` initializes as expected."""
    assert prediction_writer_callback.writing_predictions is True
    assert prediction_writer_callback.write_strategy is write_strategy
    assert prediction_writer_callback.dirpath == Path(dirpath).resolve()


def test_init_dirpath_absolute_path(prediction_writer_callback):
    """Test initialization of dirpath with absolute path."""
    absolute_path = Path("/absolute/path")
    prediction_writer_callback._init_dirpath(absolute_path)
    assert prediction_writer_callback.dirpath == absolute_path


def test_init_dirpath_relative_path(prediction_writer_callback):
    """Test initialization of dirpath with relatice path."""
    relative_path = "relative/path"
    # patch pathlib.Path.cwd to return
    mock_cwd = Path("/current/working/dir")
    with patch("pathlib.Path.cwd", return_value=mock_cwd):
        prediction_writer_callback._init_dirpath(relative_path)
        assert prediction_writer_callback.dirpath == mock_cwd / relative_path


def test_setup_prediction_directory_creation(prediction_writer_callback, dirpath):
    """
    Test prediction directory is created when `setup` is called at `stage="predict"`.
    """
    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    assert os.path.isdir(dirpath)


def test_write_on_batch_end(prediction_writer_callback):
    """
    Test `PredictionWriterCallback.write_on_batch_end`.

    Check `prediction_writer_callback.write_strategy.write_batch` is called as
    expected.
    """
    # mock write_on_batch_end inputs.
    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    prediction = Mock()
    batch_indices = [0, 1, 2]
    batch = Mock()
    batch_idx = 0
    dataloader_idx = 0

    # Mocking the dataset to be of type IterablePredDataset
    mock_dataset = Mock(spec=IterablePredDataset)
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    prediction_writer_callback.write_on_batch_end(
        trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    )
    # The different strategy functions are tested individually
    #   so they do not need to be tested here
    prediction_writer_callback.write_strategy.write_batch.assert_called_once_with(
        trainer=trainer,
        pl_module=pl_module,
        prediction=prediction,
        batch_indices=batch_indices,
        batch=batch,
        batch_idx=batch_idx,
        dataloader_idx=dataloader_idx,
        dirpath=prediction_writer_callback.dirpath,
    )


def test_write_on_batch_end_writing_predictions_off(prediction_writer_callback):
    """
    Test that `write_batch` acts as expected when writing predictions is set to off.

    Ensure `PredictionWriterCallback.write_strategy.write_batch` is not called when
    `PredictionWriterCallback.writing_predictions=False`.
    """
    prediction_writer_callback.writing_predictions = False

    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    prediction = Mock()
    batch_indices = [0, 1, 2]
    batch = Mock()
    batch_idx = 0
    dataloader_idx = 0

    # Mocking the dataset to be of type IterablePredDataset or IterableTiledPredDataset
    mock_dataset = Mock(spec=IterablePredDataset)
    trainer.predict_dataloaders = [Mock(spec=DataLoader)]
    trainer.predict_dataloaders[dataloader_idx].dataset = mock_dataset

    prediction_writer_callback.write_on_batch_end(
        trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    )
    prediction_writer_callback.write_strategy.write_batch.assert_not_called()


def test_from_write_func_params(write_strategy, dirpath):
    """
    Test `from_write_func_param` classmethod initializes `PredictWriterCallback`.
    """

    write_type = SupportedData.TIFF
    tiled = True
    write_func = None
    write_extension = None
    write_func_kwargs = None

    # mock create_write_strategy as it is already tested individually.
    with patch(
        "careamics.lightning.callbacks"
        ".prediction_writer_callback.prediction_writer_callback.create_write_strategy"
    ) as mock_create_write_strategy:
        write_strategy = Mock(spec=WriteStrategy)
        mock_create_write_strategy.return_value = write_strategy
        callback = PredictionWriterCallback.from_write_func_params(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            dirpath=dirpath,
        )
        mock_create_write_strategy.assert_called_once_with(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )
        assert callback.write_strategy == write_strategy
        assert callback.dirpath == Path(dirpath).resolve()
