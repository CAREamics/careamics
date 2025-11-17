"""Test PredictionWriterCallback class."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.config import Configuration
from careamics.config.configuration_factories import _create_ng_data_configuration
from careamics.config.data import NGDataConfig
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    PredictionWriterCallback,
    WriteStrategy,
)
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import N2VModule
from careamics.lightning.dataset_ng.prediction import convert_prediction


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


# TODO: smoke test with tiff (& example custom save func?)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_tiled_tiff(tmp_path, minimum_n2v_configuration):
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    cfg = Configuration(**minimum_n2v_configuration)

    # create NGDataset configuration
    dataset_cfg = _create_ng_data_configuration(
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        patch_size=cfg.data_config.patch_size,
        batch_size=cfg.data_config.batch_size,
        augmentations=cfg.data_config.transforms,
        patch_overlaps=(4, 4),
        train_dataloader_params=cfg.data_config.train_dataloader_params,
        val_dataloader_params=cfg.data_config.val_dataloader_params,
    )

    # create lightning module
    model = N2VModule(algorithm_config=cfg.algorithm_config)

    # create data module
    data = CareamicsDataModule(
        data_config=dataset_cfg,
        train_data=train_array,
        val_data=val_array,
    )

    # create prediction writer callback params
    dirpath = tmp_path / "predictions"
    predict_writer = PredictionWriterCallback(dirpath=dirpath)

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            predict_writer,
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    predict_writer.set_writing_strategy(write_type="tiff", tiled=True)
    means = data.train_dataset.input_stats.means
    stds = data.train_dataset.input_stats.stds

    pred_dataset_cfg = NGDataConfig(
        data_type="tiff",
        axes=cfg.data_config.axes,
        batch_size=4,
        patching={
            "name": "tiled",
            "patch_size": (8, 8),
            "overlaps": (2, 2),
        },
        transforms=[],
        image_means=means,
        image_stds=stds,
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # TODO save data does not have singleton channel axis?
    np.testing.assert_array_equal(
        save_data, predicted_images[0].squeeze(), verbose=True
    )


@pytest.mark.mps_gh_fail
def test_smoke_n2v_untiled_tiff(tmp_path, minimum_n2v_configuration):

    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    cfg = Configuration(**minimum_n2v_configuration)

    # create NGDataset configuration
    dataset_cfg = _create_ng_data_configuration(
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        patch_size=cfg.data_config.patch_size,
        batch_size=cfg.data_config.batch_size,
        augmentations=cfg.data_config.transforms,
        patch_overlaps=(4, 4),
        train_dataloader_params=cfg.data_config.train_dataloader_params,
        val_dataloader_params=cfg.data_config.val_dataloader_params,
    )

    # create lightning module
    model = N2VModule(algorithm_config=cfg.algorithm_config)

    # create data module
    data = CareamicsDataModule(
        data_config=dataset_cfg,
        train_data=train_array,
        val_data=val_array,
    )

    # create prediction writer callback params
    dirpath = tmp_path / "predictions"
    predict_writer = PredictionWriterCallback(dirpath=dirpath)

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            predict_writer,
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    predict_writer.set_writing_strategy(write_type="tiff", tiled=False)
    means = data.train_dataset.input_stats.means
    stds = data.train_dataset.input_stats.stds

    pred_dataset_cfg = NGDataConfig(
        data_type="tiff",
        axes=cfg.data_config.axes,
        batch_size=4,
        patching={
            "name": "whole",
        },
        transforms=[],
        image_means=means,
        image_stds=stds,
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images = convert_prediction(predicted, tiled=False)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # save data has singleton channel axis
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)


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
