"""Test PredictionWriterCallback class."""

import os
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import pytest
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer

from careamics.config.support import SupportedData
from careamics.file_io import WriteFunc
from careamics.lightning.callbacks import PredictionWriterCallback
from careamics.lightning.callbacks import prediction_writer_callback
from careamics.lightning.callbacks.prediction_writer_callback import WriteStrategy
from careamics.dataset import IterablePredDataset

# TODO: add docs

@pytest.fixture
def write_strategy():
    return Mock(spec=WriteStrategy)


@pytest.fixture
def dirpath(tmp_path: Path):
    return tmp_path / "predictions"


@pytest.fixture
def prediction_writer_callback(
    write_strategy: WriteStrategy, dirpath: Union[Path, str]
):
    return PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath)


def test_initialization(prediction_writer_callback, write_strategy, dirpath):
    assert prediction_writer_callback.writing_predictions is True
    assert prediction_writer_callback.write_strategy is write_strategy
    assert prediction_writer_callback.dirpath == Path(dirpath).resolve()


def test_init_dirpath_absolute_path(prediction_writer_callback):
    absolute_path = Path("/absolute/path")
    prediction_writer_callback._init_dirpath(absolute_path)
    assert prediction_writer_callback.dirpath == absolute_path


def test_init_dirpath_relative_path(prediction_writer_callback):
    relative_path = "relative/path"
    with patch("pathlib.Path.cwd", return_value=Path("/current/working/dir")):
        prediction_writer_callback._init_dirpath(relative_path)
        assert prediction_writer_callback.dirpath == Path(
            "/current/working/dir/relative/path"
        )


def test_setup_prediction_directory_creation(prediction_writer_callback, dirpath):

    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    assert os.path.isdir(dirpath)


def test_write_on_batch_end(prediction_writer_callback):
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

    prediction_writer_callback.write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
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

    prediction_writer_callback.write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
    prediction_writer_callback.write_strategy.write_batch.assert_not_called()

def test_from_write_func_params(write_strategy, dirpath):
    write_type = SupportedData.TIFF
    tiled = True
    write_func = None
    write_extension = None
    write_func_kwargs = None

    with (
        patch(
        "careamics.lightning.callbacks"
        ".prediction_writer_callback.prediction_writer_callback.create_write_strategy"
        ) as mock_create_write_strategy
    ):
        write_strategy = Mock(spec=WriteStrategy)
        mock_create_write_strategy.return_value = write_strategy
        callback = PredictionWriterCallback.from_write_func_params(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
            dirpath=dirpath
        )
        mock_create_write_strategy.assert_called_once_with(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs
        )
        assert callback.write_strategy == write_strategy
        assert callback.dirpath == Path(dirpath).resolve()



