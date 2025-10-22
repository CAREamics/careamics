from pathlib import Path

import numpy as np
import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)

from careamics.config import Configuration
from careamics.config.configuration_factories import create_pn2v_configuration
from careamics.lightning import (
    create_careamics_module,
    create_predict_datamodule,
    create_train_datamodule,
)
from careamics.lightning.lightning_module import FCNModule
from careamics.prediction_utils import convert_outputs, convert_outputs_pn2v

pytestmark = pytest.mark.mps_gh_fail


def test_smoke_n2v_2d_array(tmp_path, minimum_n2v_configuration):
    """Test a full run of N2V training with the lightning API."""
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    cfg = Configuration(**minimum_n2v_configuration)

    # create lightning module
    model = create_careamics_module(
        algorithm=cfg.algorithm_config.algorithm,
        loss=cfg.algorithm_config.loss,
        architecture=cfg.algorithm_config.model.architecture,
    )

    # create data module
    data = create_train_datamodule(
        train_data=train_array,
        val_data=val_array,
        data_type=cfg.data_config.data_type,
        patch_size=cfg.data_config.patch_size,
        axes=cfg.data_config.axes,
        batch_size=cfg.data_config.batch_size,
    )

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            )
        ],
    )

    # train
    trainer.fit(model, datamodule=data)
    assert Path(tmp_path / "checkpoints" / "test_lightning_api.ckpt").exists()

    # predict
    means, stds = data.get_data_statistics()
    predict_data = create_predict_datamodule(
        pred_data=val_array,
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        image_means=means,
        image_stds=stds,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    assert predicted[0].squeeze().shape == val_array.shape


def test_smoke_n2v_2d_tiling(tmp_path, minimum_n2v_configuration):
    """Test a full run of N2V training with the lightning API and tiled prediction."""
    # training data
    rng = np.random.default_rng(42)
    train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
    val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

    cfg = Configuration(**minimum_n2v_configuration)

    # create lightning module
    model = create_careamics_module(
        algorithm=cfg.algorithm_config.algorithm,
        loss=cfg.algorithm_config.loss,
        architecture=cfg.algorithm_config.model.architecture,
    )

    # create data module
    data = create_train_datamodule(
        train_data=train_array,
        val_data=val_array,
        data_type=cfg.data_config.data_type,
        patch_size=cfg.data_config.patch_size,
        axes=cfg.data_config.axes,
        batch_size=cfg.data_config.batch_size,
    )

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            )
        ],
    )

    # train
    trainer.fit(model, datamodule=data)
    assert Path(tmp_path / "checkpoints" / "test_lightning_api.ckpt").exists()

    # predict
    means, stds = data.get_data_statistics()
    predict_data = create_predict_datamodule(
        pred_data=val_array,
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        image_means=means,
        image_stds=stds,
        tile_size=(8, 8),
        tile_overlap=(2, 2),
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_stitched = convert_outputs(predicted, tiled=True)
    assert predicted_stitched[0].shape[-2:] == val_array.shape


def test_smoke_pn2v_2d_array(tmp_path, create_dummy_noise_model):
    """Test a full run of PN2V training with the lightning API."""
    # Create dummy noise model file
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, (128, 128)).astype(np.float32)
    val_array = rng.integers(0, 255, (128, 128)).astype(np.float32)

    # create PN2V configuration
    cfg = create_pn2v_configuration(
        experiment_name="test_pn2v",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=2,
        nm_path=str(tmp_path / "dummy_noise_model.npz"),
        num_epochs=1,
    )

    # create lightning module
    model = FCNModule(cfg.algorithm_config)

    # create data module
    data = create_train_datamodule(
        train_data=train_array,
        val_data=val_array,
        data_type=cfg.data_config.data_type,
        patch_size=cfg.data_config.patch_size,
        axes=cfg.data_config.axes,
        batch_size=cfg.data_config.batch_size,
    )

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_pn2v_lightning_api",
            )
        ],
    )

    # train
    trainer.fit(model, datamodule=data)
    assert Path(tmp_path / "checkpoints" / "test_pn2v_lightning_api.ckpt").exists()

    # predict
    means, stds = data.get_data_statistics()
    predict_data = create_predict_datamodule(
        pred_data=val_array,
        data_type=cfg.data_config.data_type,
        axes=cfg.data_config.axes,
        image_means=means,
        image_stds=stds,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_avg, mse = convert_outputs_pn2v(predicted, tiled=False)
    assert predicted_avg[0].squeeze().shape == val_array.shape
    assert mse[0].squeeze().shape == val_array.shape
