"""Tests for the COSDD Lightning module."""

from pathlib import Path

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer

from careamics.config import COSDDConfiguration, create_advanced_cosdd_config
from careamics.config.data.normalization_config import MeanStdConfig
from careamics.lightning.data import CareamicsDataModule
from careamics.lightning.modules import COSDDModule, create_module
from careamics.lightning.utils.load_checkpoint import (
    load_config_from_checkpoint,
    load_module_from_checkpoint,
)


def _config(**kwargs) -> COSDDConfiguration:
    """Small COSDD configuration with known normalization statistics."""
    params = {
        "experiment_name": "test_cosdd",
        "data_type": "array",
        "axes": "SYX",
        "patch_size": [16, 16],
        "batch_size": 2,
        "n_val_patches": 2,
        "n_grad_batches": 1,
        "num_workers": 0,
        "normalization": MeanStdConfig(input_means=[0.0], input_stds=[1.0]),
        "model_params": {"s_code_channels": 8, "n_layers": 2, "checkpointed": False},
    }
    params.update(kwargs)
    return create_advanced_cosdd_config(**params)


def _trainer(**kwargs) -> Trainer:
    params = {
        "max_epochs": 1,
        "accelerator": "cpu",
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    params.update(kwargs)
    return Trainer(**params)


@pytest.fixture
def train_array() -> np.ndarray:
    """Noisy images with noise correlated along X."""
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(4, 32, 32)).astype(np.float32)
    return noise + np.roll(noise, 1, axis=-1)


def test_create_module():
    """Test creating the module from a configuration or a dictionary."""
    config = _config()

    module = create_module(config.algorithm_config)
    assert isinstance(module, COSDDModule)

    module = COSDDModule(config.algorithm_config.model_dump())
    assert module.config == config.algorithm_config

    with pytest.raises(TypeError):
        COSDDModule(config.data_config)  # type: ignore[arg-type]


@pytest.mark.parametrize("use_direct_denoiser", [True, False])
def test_configure_optimizers(use_direct_denoiser):
    """Test that each network gets its own optimizer and scheduler."""
    module = COSDDModule(
        _config(use_direct_denoiser=use_direct_denoiser).algorithm_config
    )

    optimizers, schedulers = module.configure_optimizers()
    assert len(optimizers) == len(schedulers) == (3 if use_direct_denoiser else 2)

    # all trainable parameters are optimized, exactly once
    optimized = [p for opt in optimizers for g in opt.param_groups for p in g["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in module.parameters()}


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("use_direct_denoiser", [True, False])
def test_training(train_array, use_direct_denoiser):
    """Test training with the CAREamics data module and the Lightning API."""
    config = _config(use_direct_denoiser=use_direct_denoiser)
    module = COSDDModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_array
    )
    trainer = _trainer()

    trainer.fit(module, datamodule=datamodule)

    metrics = trainer.callback_metrics
    expected = {"train_loss_epoch", "train_kl", "train_nll", "train_sd_loss"}
    expected |= {"val_loss", "val_kl", "val_nll", "val_sd_loss"}
    if use_direct_denoiser:
        expected |= {"train_dd_loss", "val_dd_loss"}
    assert expected <= set(metrics)
    assert all(torch.isfinite(metrics[name]) for name in expected)
    assert ("val_dd_loss" in metrics) == use_direct_denoiser

    # ReduceLROnPlateau schedulers were stepped with the validation losses
    for scheduler in module.lr_schedulers():
        assert np.isfinite(scheduler.best)


@pytest.mark.mps_gh_fail
def test_training_and_prediction_checkpointed(train_array):
    """Test training and prediction with activation checkpointing (the default)."""
    config = _config(
        model_params={"s_code_channels": 8, "n_layers": 2, "checkpointed": True}
    )
    assert config.algorithm_config.model.checkpointed
    module = COSDDModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_array
    )
    trainer = _trainer(limit_train_batches=2)

    trainer.fit(module, datamodule=datamodule)
    assert torch.isfinite(trainer.callback_metrics["train_loss_epoch"])

    module.n_samples = 2
    pred_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(data_config=pred_config, pred_data=train_array)
    outputs = trainer.predict(module, datamodule=datamodule)

    assert outputs is not None
    for prediction, uncertainty in outputs:
        assert prediction.data.shape == (2, 1, 32, 32)
        assert np.isfinite(prediction.data).all()
        assert uncertainty is not None


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize("n_grad_batches, n_updates", [(1, 3), (2, 2), (3, 1), (4, 1)])
def test_gradient_accumulation(train_array, n_grad_batches, n_updates):
    """Test that weights are updated every `n_grad_batches` and on the last batch.

    Updating on the last batch of the epoch ensures that no gradient is carried over
    to the next epoch.
    """
    config = _config(n_grad_batches=n_grad_batches)
    module = COSDDModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_array
    )
    trainer = _trainer(limit_train_batches=3, limit_val_batches=0)

    trainer.fit(module, datamodule=datamodule)

    # Lightning counts the step of each of the 3 optimizers (direct denoiser)
    assert trainer.global_step == 3 * n_updates
    assert all(p.grad is None for p in module.parameters())


@pytest.mark.mps_gh_fail
def test_plateau_schedulers_step_after_validation(train_array):
    """Test that `ReduceLROnPlateau` schedulers only step in epochs with validation.

    They must not step on the validation losses of a previous epoch.
    """
    config = _config()
    module = COSDDModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_array
    )
    trainer = _trainer(
        max_epochs=3,
        check_val_every_n_epoch=2,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(module, datamodule=datamodule)

    schedulers = module.lr_schedulers()
    assert len(schedulers) == 3
    for scheduler in schedulers:
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.last_epoch == 1


@pytest.mark.parametrize("use_direct_denoiser", [True, False])
@pytest.mark.parametrize("n_samples", [1, 3])
def test_prediction(train_array, use_direct_denoiser, n_samples):
    """Test prediction with the direct denoiser or by averaging samples."""
    config = _config(use_direct_denoiser=use_direct_denoiser)
    module = COSDDModule(config.algorithm_config)
    module.n_samples = n_samples
    pred_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(data_config=pred_config, pred_data=train_array)

    outputs = _trainer().predict(module, datamodule=datamodule)

    assert outputs is not None
    for prediction, uncertainty in outputs:
        assert prediction.data.shape == (2, 1, 32, 32)
        if n_samples == 1:
            assert uncertainty is None
        else:
            assert uncertainty.data.shape == prediction.data.shape
            assert np.all(uncertainty.data > 0)


def test_direct_prediction_is_deterministic(train_array):
    """Test that the direct denoiser prediction is deterministic."""
    config = _config()
    module = COSDDModule(config.algorithm_config)
    pred_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(data_config=pred_config, pred_data=train_array)
    trainer = _trainer()

    first = trainer.predict(module, datamodule=datamodule)
    second = trainer.predict(module, datamodule=datamodule)

    assert first is not None and second is not None
    np.testing.assert_array_equal(first[0][0].data, second[0][0].data)


def test_direct_prediction_requires_direct_denoiser(train_array):
    """Test that the direct prediction fails without a direct denoiser."""
    config = _config(use_direct_denoiser=False)
    config.algorithm_config.direct_prediction = True
    module = COSDDModule(config.algorithm_config)
    pred_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(data_config=pred_config, pred_data=train_array)

    with pytest.raises(ValueError, match="direct denoiser"):
        _trainer().predict(module, datamodule=datamodule)


def test_prediction_without_direct_prediction(train_array):
    """Test that disabling direct prediction draws a signal sample."""
    config = _config()
    module = COSDDModule(config.algorithm_config)
    module.config.direct_prediction = False
    pred_config = config.data_config.convert_mode("predicting")
    datamodule = CareamicsDataModule(data_config=pred_config, pred_data=train_array)
    trainer = _trainer()

    first = trainer.predict(module, datamodule=datamodule)
    second = trainer.predict(module, datamodule=datamodule)

    assert first is not None and second is not None
    prediction, uncertainty = first[0]
    assert prediction.data.shape == (2, 1, 32, 32)
    assert uncertainty is None
    assert not np.array_equal(prediction.data, second[0][0].data)


@pytest.mark.parametrize(
    "use_direct_denoiser, direct_prediction, deterministic",
    [
        (True, None, True),
        (True, True, True),
        (True, False, False),
        (False, None, False),
    ],
)
def test_forward(use_direct_denoiser, direct_prediction, deterministic):
    """Test that `forward` follows `direct_prediction`."""
    config = _config(use_direct_denoiser=use_direct_denoiser)
    config.algorithm_config.direct_prediction = direct_prediction
    module = COSDDModule(config.algorithm_config).eval()
    x = torch.randn(2, 1, 24, 24)

    with torch.no_grad():
        first, second = module(x), module(x)

    assert first.shape == x.shape
    assert torch.equal(first, second) == deterministic


def test_forward_requires_direct_denoiser():
    """Test that `forward` fails with direct prediction and no direct denoiser."""
    config = _config(use_direct_denoiser=False)
    config.algorithm_config.direct_prediction = True
    module = COSDDModule(config.algorithm_config)

    with pytest.raises(ValueError, match="direct denoiser"):
        module(torch.randn(1, 1, 16, 16))


@pytest.mark.parametrize("noise_direction", ["x", "none"])
def test_generation(noise_direction):
    """Test generating images from the prior and reconstructing images."""
    module = COSDDModule(_config(noise_direction=noise_direction).algorithm_config)
    module.train()

    generated = module.sample_prior(2)
    assert generated["x"].shape == generated["s"].shape == (2, 1, 16, 16)
    assert generated["x"].std() > 0

    reconstructed = module.reconstruct(torch.randn(2, 1, 24, 24))
    assert reconstructed["x_hat"].shape == (2, 1, 24, 24)
    assert reconstructed["s_hat"].shape == (2, 1, 24, 24)
    assert reconstructed["x_hat"].std() > 0

    # the training mode is restored
    assert all(m.training for m in module.modules())


@pytest.mark.mps_gh_fail
def test_checkpoint(train_array, tmp_path: Path):
    """Test loading the module and configuration from a checkpoint."""
    config = _config()
    module = COSDDModule(config.algorithm_config)
    datamodule = CareamicsDataModule(
        data_config=config.data_config, train_data=train_array
    )
    trainer = _trainer()
    trainer.fit(module, datamodule=datamodule)
    ckpt_path = tmp_path / "cosdd.ckpt"
    trainer.save_checkpoint(ckpt_path)

    loaded_module = load_module_from_checkpoint(ckpt_path)
    assert isinstance(loaded_module, COSDDModule)
    for a, b in zip(
        module.state_dict().values(),
        loaded_module.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b)

    loaded_config = load_config_from_checkpoint(ckpt_path)
    assert isinstance(loaded_config, COSDDConfiguration)
    assert loaded_config.algorithm_config == config.algorithm_config
