"""Test PredictionWriterCallback class."""

from pathlib import Path

import numpy as np
import pytest
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data.dataloader import default_collate

from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.callbacks.prediction import (
    ImageWriteStrategy,
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
    assert prediction_writer_callback.is_enabled is True
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
    # `setup` reads `trainer.datamodule` to preserve the source directory structure;
    # None mimics predicting from raw dataloaders (no datamodule provided)
    trainer.datamodule = None
    pl_module = mocker.Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    assert Path(dirpath).is_dir()


def test_write_on_batch_end_writing_predictions_off(mocker, prediction_writer_callback):
    """
    Test that `write_batch` acts as expected when writing predictions is off.

    Ensure `PredictionWriterCallback.write_strategy.write_batch` is not called when
    `PredictionWriterCallback.writing_predictions=False`.
    """
    prediction_writer_callback.is_enabled = False
    write_strategy = mocker.Mock(spec=WriteStrategy)
    prediction_writer_callback.write_strategy = write_strategy

    prediction_writer_callback.write_on_batch_end(
        None, None, [], None, None, None, None
    )
    write_strategy.write_batch.assert_not_called()


def _uncertainty_batch() -> tuple[ImageRegionData, ImageRegionData]:
    prediction = ImageRegionData(
        data=np.ones((1, 8, 8), dtype=np.float32),
        source="x.tiff",
        data_shape=(1, 8, 8),
        dtype=str(np.float32),
        axes="CYX",
        target_axes="CYX",
        original_data_shape=(8, 8),
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (0, 0),
            "patch_size": (8, 8),
        },
        additional_metadata={},
    )
    uncertainty = prediction._replace(data=np.full((1, 8, 8), 0.5, dtype=np.float32))
    return default_collate([prediction]), default_collate([uncertainty])


def test_write_on_batch_end_skips_uncertainty_by_default(
    mocker,
    dirpath: Path,
):
    """Test that uncertainty is not written by default."""
    callback = PredictionWriterCallback(dirpath=dirpath)
    callback.set_writing_strategy(write_type="tiff", tiled=False)
    assert callback.uncertainty_write_strategy is None
    write_strategy = mocker.Mock(spec=WriteStrategy)
    callback.write_strategy = write_strategy

    pred, unc = _uncertainty_batch()
    callback.write_on_batch_end(None, None, (pred, unc), None, pred, 0, 0)

    write_strategy.write_batch.assert_called_once()


def test_write_on_batch_end_writes_uncertainty_when_enabled(mocker, dirpath: Path):
    """Test that uncertainty is written by its own write strategy."""
    pred_strategy = mocker.Mock(spec=WriteStrategy)
    unc_strategy = mocker.Mock(spec=WriteStrategy)
    callback = PredictionWriterCallback(
        dirpath=dirpath,
        write_strategy=pred_strategy,
        uncertainty_write_strategy=unc_strategy,
    )

    pred, unc = _uncertainty_batch()
    callback.write_on_batch_end(None, None, (pred, unc), None, pred, 0, 0)

    pred_strategy.write_batch.assert_called_once()
    unc_strategy.write_batch.assert_called_once()


def test_set_writing_strategy_builds_uncertainty_strategy(dirpath: Path):
    """Test that set_writing_strategy builds an uncertainty write strategy."""
    callback = PredictionWriterCallback(dirpath=dirpath)
    callback.set_writing_strategy(
        write_type="tiff", tiled=False, write_uncertainty=True
    )

    assert isinstance(callback.write_strategy, ImageWriteStrategy)
    assert callback.write_strategy.filename_postfix == ""
    uncertainty_strategy = callback.uncertainty_write_strategy
    assert isinstance(uncertainty_strategy, ImageWriteStrategy)
    assert uncertainty_strategy.filename_postfix == "_uncertainty"


def test_set_writing_strategy_zarr_uncertainty_not_implemented(dirpath: Path):
    """Test that writing uncertainty to zarr raises an error."""
    callback = PredictionWriterCallback(dirpath=dirpath)
    with pytest.raises(NotImplementedError, match="tiff"):
        callback.set_writing_strategy(
            write_type="zarr", tiled=True, write_uncertainty=True
        )
