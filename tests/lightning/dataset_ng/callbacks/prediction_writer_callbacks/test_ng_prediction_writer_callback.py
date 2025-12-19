"""Test PredictionWriterCallback class."""

from pathlib import Path

import pytest
from pytorch_lightning import LightningModule, Trainer

from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    PredictionWriterCallback,
    WriteStrategy,
)


@pytest.fixture
def write_strategy(mocker):
    """Mock `WriteFunc`."""
    return mocker.Mock(spec=WriteStrategy)


@pytest.fixture
def dirpath(tmp_path: Path):
    """Directory path."""
    return tmp_path / "predictions"


@pytest.fixture
def prediction_writer_callback(write_strategy: WriteStrategy, dirpath: Path | str):
    """Initialized `PredictionWriterCallback`."""
    return PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath)


def test_initialization(prediction_writer_callback, write_strategy, dirpath):
    """Test `PredictionWriterCallback` initializes as expected."""
    assert prediction_writer_callback.writing_predictions is True
    assert prediction_writer_callback.write_strategy is write_strategy
    assert prediction_writer_callback.dirpath == Path(dirpath).resolve()


def test_init_dirpath_absolute_path(prediction_writer_callback):
    """Test initialization of dirpath with absolute path."""
    absolute_path = Path("/absolute/path").absolute()
    prediction_writer_callback._init_dirpath(absolute_path)
    assert prediction_writer_callback.dirpath == absolute_path


def test_init_dirpath_relative_path(mocker, prediction_writer_callback):
    """Test initialization of dirpath with relative path."""
    relative_path = "relative/path"
    # patch pathlib.Path.cwd to return
    mock_cwd = Path("/current/working/dir")
    with mocker.patch("pathlib.Path.cwd", return_value=mock_cwd):
        prediction_writer_callback._init_dirpath(relative_path)
        assert prediction_writer_callback.dirpath == mock_cwd / relative_path


def test_setup_prediction_directory_creation(
    mocker, prediction_writer_callback, dirpath
):
    """
    Test prediction directory is created when `setup` is called at `stage="predict"`.
    """
    trainer = mocker.Mock(spec=Trainer)
    pl_module = mocker.Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    assert Path(dirpath).is_dir()


def test_write_on_batch_end_writing_predictions_off(mocker, prediction_writer_callback):
    """
    Test that `write_batch` acts as expected when writing predictions is set to off.

    Ensure `PredictionWriterCallback.write_strategy.write_batch` is not called when
    `PredictionWriterCallback.writing_predictions=False`.
    """
    prediction_writer_callback.writing_predictions = False
    write_strategy = mocker.Mock(spec=WriteStrategy)
    prediction_writer_callback.write_strategy = write_strategy

    prediction_writer_callback.write_on_batch_end(
        None, None, [], None, None, None, None
    )
    write_strategy.write_batch.assert_not_called()
