"""Lightning training checkpoints opened through CAREamist.

Issue #1108: a checkpoint written by a Lightning Trainer, without
``ConfigSaverCallback``, must keep the normalization statistics computed during
``fit`` and must predict through ``CAREamist``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics.careamist import CAREamist
from careamics.config.factories import (
    create_advanced_microsplit_config,
    create_advanced_n2v_config,
)
from careamics.lightning import CareamicsDataModule, N2VModule
from careamics.lightning.modules.microsplit_module import MicroSplitModule


def _trainer(tmp_path: Path) -> tuple[Trainer, ModelCheckpoint]:
    """Return a Trainer whose only checkpoint callback is ModelCheckpoint."""
    checkpoint = ModelCheckpoint(
        dirpath=tmp_path / "checkpoints", save_last=True, save_top_k=0
    )
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint],
    )
    return trainer, checkpoint


def test_n2v_lightning_checkpoint_predicts_through_careamist(tmp_path: Path) -> None:
    """Test an N2V Trainer checkpoint predicts through CAREamist with its stats.

    Statistics are left unset in the configuration, so they are computed during
    ``fit``. The Trainer has no ``ConfigSaverCallback``.
    """
    rng = np.random.default_rng(0)
    train = rng.random((32, 32)).astype(np.float32)
    val = rng.random((32, 32)).astype(np.float32)
    config = create_advanced_n2v_config(
        experiment_name="n2v_lightning",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        num_workers=0,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    assert config.data_config.normalization.input_means is None

    module = N2VModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        config.data_config, train_data=train, val_data=val
    )
    trainer, checkpoint = _trainer(tmp_path)
    trainer.fit(module, datamodule=datamodule)

    saved = torch.load(
        checkpoint.last_model_path, map_location="cpu", weights_only=False
    )
    saved_means = saved["datamodule_hyper_parameters"]["data_config"]["normalization"][
        "input_means"
    ]
    assert "careamics_info" not in saved
    assert saved_means is not None

    careamist = CAREamist(checkpoint_path=checkpoint.last_model_path, work_dir=tmp_path)
    predicted, _ = careamist.predict(val)

    np.testing.assert_allclose(
        careamist.config.data_config.normalization.input_means, saved_means
    )
    assert np.asarray(predicted[0]).shape == val.shape


def test_n2v_checkpoint_saved_after_predict_keeps_training_stats(
    tmp_path: Path,
) -> None:
    """Test a checkpoint saved after prediction still has the training statistics.

    Prediction replaces the trainer's datamodule. The saved checkpoint must keep
    the training configuration, including statistics computed during ``fit``.
    """
    rng = np.random.default_rng(0)
    train = rng.random((32, 32)).astype(np.float32)
    val = rng.random((32, 32)).astype(np.float32)
    config = create_advanced_n2v_config(
        experiment_name="n2v_after_predict",
        data_type="array",
        axes="YX",
        patch_size=(8, 8),
        batch_size=2,
        num_epochs=1,
        num_workers=0,
        roi_size=5,
        masked_pixel_percentage=5,
    )
    module = N2VModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        config.data_config, train_data=train, val_data=val
    )
    trainer, _checkpoint = _trainer(tmp_path)
    trainer.fit(module, datamodule=datamodule)
    training_means = datamodule.hparams["data_config"]["normalization"]["input_means"]

    trainer.predict(
        module,
        datamodule=CareamicsDataModule(
            config.data_config.convert_mode("predicting"), pred_data=val
        ),
    )
    after_predict = tmp_path / "after_predict.ckpt"
    trainer.save_checkpoint(after_predict)

    careamist = CAREamist(
        checkpoint_path=after_predict, work_dir=tmp_path / "careamist"
    )
    predicted, _ = careamist.predict(val)

    assert careamist.config.data_config.mode == "training"
    np.testing.assert_allclose(
        careamist.config.data_config.normalization.input_means, training_means
    )
    assert np.asarray(predicted[0]).shape == val.shape


@pytest.mark.lvae
def test_microsplit_lightning_checkpoint_predicts_through_careamist(
    tmp_path: Path,
) -> None:
    """Test a MicroSplit Trainer checkpoint predicts through CAREamist with its stats.

    Statistics are left unset in the configuration, so they are computed during
    ``fit``. The Trainer has no ``ConfigSaverCallback``.
    """
    rng = np.random.default_rng(1)
    target = rng.random((2, 2, 64, 64)).astype(np.float32)
    train = target.sum(axis=1, keepdims=True) + rng.normal(0, 0.05, (2, 1, 64, 64))
    train = train.astype(np.float32)
    config = create_advanced_microsplit_config(
        experiment_name="microsplit_lightning",
        data_type="array",
        axes="SCYX",
        patch_size=[64, 64],
        batch_size=2,
        output_channels=2,
        num_epochs=1,
        num_steps=1,
        multiscale_count=1,
        augmentations=[],
        gaussian_likelihood_weight=1.0,
        noise_model_likelihood_weight=0.0,
        model_params={"z_dims": [32, 32], "n_filters": 8},
        num_workers=0,
        seed=1,
    )
    assert config.data_config.normalization.input_means is None

    module = MicroSplitModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        config.data_config,
        train_data=train,
        train_data_target=target,
        val_data=train,
        val_data_target=target,
    )
    trainer, checkpoint = _trainer(tmp_path)
    trainer.fit(module, datamodule=datamodule)

    saved = torch.load(
        checkpoint.last_model_path, map_location="cpu", weights_only=False
    )
    saved_norm = saved["datamodule_hyper_parameters"]["data_config"]["normalization"]
    assert "careamics_info" not in saved
    assert saved_norm["input_means"] is not None
    assert len(saved_norm["target_means"]) == 2

    careamist = CAREamist(checkpoint_path=checkpoint.last_model_path, work_dir=tmp_path)
    predicted, _ = careamist.predict(
        train, tile_size=(64, 64), tile_overlap=(32, 32)
    )

    np.testing.assert_allclose(
        careamist.config.data_config.normalization.input_means,
        saved_norm["input_means"],
    )
    np.testing.assert_allclose(
        careamist.config.data_config.normalization.target_means,
        saved_norm["target_means"],
    )
    assert np.asarray(predicted[0]).shape == (2, 2, 64, 64)
