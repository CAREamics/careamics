"""Test PredictionWriterCallback class."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.config import Configuration, N2VAlgorithm
from careamics.config.configuration_factories import create_ng_data_configuration
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
    dataset_cfg = create_ng_data_configuration(
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        patch_size=cfg.data_config.patch_size,
        batch_size=cfg.data_config.batch_size,
        augmentations=cfg.data_config.transforms,
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

    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="tiff",
        new_batch_size=4,
        new_patch_size=(8, 8),
        overlap_size=(2, 2),
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # predicted image is a list
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_tiled_tiff_channels(tmp_path):
    rng = np.random.default_rng(42)

    channels = [0, 3]

    # training data
    train_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    algorithm_config = N2VAlgorithm(
        model={
            "architecture": "UNet",
            "in_channels": len(channels),
            "num_classes": len(channels),
        }
    )

    # create NGDataset configuration
    dataset_cfg: NGDataConfig = create_ng_data_configuration(
        data_type="array",
        axes="CYX",
        patch_size=(16, 16),
        batch_size=2,
        channels=channels,
    )

    # create lightning module
    model = N2VModule(algorithm_config=algorithm_config)

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
    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="tiff",
        new_batch_size=4,
        new_patch_size=(8, 8),
        overlap_size=(2, 2),
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # predicted images is a list
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)
    assert save_data.shape[0] == len(channels)


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
    dataset_cfg = create_ng_data_configuration(
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        patch_size=cfg.data_config.patch_size,
        batch_size=cfg.data_config.batch_size,
        augmentations=cfg.data_config.transforms,
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
    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="tiff",
        new_batch_size=4,
    )

    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=False)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # predicted images is a list
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_untiled_tiff_channels(tmp_path, minimum_n2v_configuration):

    rng = np.random.default_rng(42)
    channels = [0, 3]

    # training data
    train_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    algorithm_config = N2VAlgorithm(
        model={
            "architecture": "UNet",
            "in_channels": len(channels),
            "num_classes": len(channels),
        }
    )

    # create NGDataset configuration
    dataset_cfg: NGDataConfig = create_ng_data_configuration(
        data_type="array",
        axes="CYX",
        patch_size=(16, 16),
        batch_size=2,
        channels=channels,
    )

    # create lightning module
    model = N2VModule(algorithm_config=algorithm_config)

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
    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="tiff",
        new_batch_size=4,
    )

    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=False)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # predicted images is a list
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)
    assert save_data.shape[0] == len(channels)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_tiled_zarr(tmp_path, minimum_n2v_configuration):

    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    # write train array to zarr
    z = zarr.open(tmp_path / "train.zarr", mode="w")
    group = z.create_group("data")
    array = group.create_array(name="single_image", data=train_array, chunks=(16, 16))
    path_to_array = str(array.store_path)

    cfg = Configuration(**minimum_n2v_configuration)

    # create NGDataset configuration
    dataset_cfg = create_ng_data_configuration(
        data_type="array",
        axes=cfg.data_config.axes,
        patch_size=cfg.data_config.patch_size,
        batch_size=cfg.data_config.batch_size,
        augmentations=cfg.data_config.transforms,
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
    predict_writer.set_writing_strategy(write_type="zarr", tiled=True)

    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="zarr",
        new_batch_size=4,
        new_patch_size=(8, 8),
        overlap_size=(2, 2),
        new_in_memory=False,
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=path_to_array,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    z_out = zarr.open(dirpath / "train_output.zarr")
    array_output = z_out["data"]["single_image"]

    np.testing.assert_array_equal(array_output, predicted_images[0], verbose=True)


@pytest.mark.mps_gh_fail
def test_smoke_n2v_tiled_zarr_channels(tmp_path, minimum_n2v_configuration):

    rng = np.random.default_rng(42)

    channels = [0, 3]

    # training data
    train_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (5, 32, 32)).astype(np.float32)

    # write train array to zarr
    z = zarr.open(tmp_path / "train.zarr", mode="w")
    group = z.create_group("data")
    array = group.create_array(
        name="single_image", data=train_array, chunks=(1, 16, 16)
    )
    path_to_array = str(array.store_path)

    algorithm_config = N2VAlgorithm(
        model={
            "architecture": "UNet",
            "in_channels": len(channels),
            "num_classes": len(channels),
        }
    )

    # create NGDataset configuration
    dataset_cfg = create_ng_data_configuration(
        data_type="array",
        axes="CYX",
        patch_size=(16, 16),
        batch_size=2,
        channels=channels,
    )

    # create lightning module
    model = N2VModule(algorithm_config=algorithm_config)

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
    predict_writer.set_writing_strategy(write_type="zarr", tiled=True)

    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="zarr",
        new_batch_size=4,
        new_patch_size=(8, 8),
        overlap_size=(2, 2),
        new_in_memory=False,
    )
    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=path_to_array,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    z_out = zarr.open(dirpath / "train_output.zarr")
    array_output = z_out["data"]["single_image"]

    np.testing.assert_array_equal(array_output, predicted_images[0], verbose=True)
    assert array_output.shape[0] == len(channels)


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
