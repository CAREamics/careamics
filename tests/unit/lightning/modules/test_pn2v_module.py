"""Tests for the PN2V Lightning module."""

from pathlib import Path

import numpy as np
import pytest
import torch

from careamics import CAREamist
from careamics.config.factories import create_advanced_pn2v_config
from careamics.lightning.modules import PN2VModule, create_module
from careamics.lightning.modules.pn2v_module import pn2v_mmse
from careamics.models.lvae.noise_models import MultiChannelNoiseModel
from tests.utils import gaussian_noise_model_testing


def _pn2v_config(tmp_name: str = "pn2v_module", **kwargs):
    """Return a small PN2V configuration for training on `SYX` arrays."""
    params = {
        "experiment_name": tmp_name,
        "data_type": "array",
        "axes": "SYX",
        "patch_size": (16, 16),
        "batch_size": 2,
        "num_epochs": 1,
        "num_steps": 1,
        "num_classes": 2,
        "model_params": {
            "depth": 2,
            "num_channels_init": 8,
            "use_batch_norm": False,
        },
        "n_val_patches": 1,
        "masked_pixel_percentage": 5,
        "roi_size": 5,
    }
    params.update(kwargs)
    return create_advanced_pn2v_config(**params)


def test_likelihood_broadcasts_over_sample_channels(
    pn2v_noise_model: MultiChannelNoiseModel,
):
    """Observations with one channel score every prior sample."""
    observations = torch.rand(2, 1, 8, 8)
    samples = torch.rand(2, 4, 8, 8)

    likelihoods = pn2v_noise_model.nmodel_0.likelihood(observations, samples)

    assert likelihoods.shape == samples.shape


def test_pn2v_mmse_collapses_sample_axis(pn2v_noise_model: MultiChannelNoiseModel):
    """MMSE is a weighted mean over the sample channel."""
    observations = torch.full((1, 1, 4, 4), 1.0)
    samples = torch.rand(1, 3, 4, 4)

    estimate = pn2v_mmse(samples, observations, pn2v_noise_model.nmodel_0)

    assert estimate.shape == (1, 1, 4, 4)
    assert torch.isfinite(estimate).all()


def test_create_module_returns_pn2v_module():
    """PN2V configs build a PN2VModule with the configured sample scale."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_module",
        data_type="array",
        axes="YX",
        patch_size=[32, 32],
        batch_size=1,
        num_classes=2,
        model_params={"depth": 2, "num_channels_init": 8},
    )

    module = create_module(config.algorithm_config)

    assert isinstance(module, PN2VModule)
    assert module.config.sample_scale == 10.0
    output = module(torch.rand(1, 1, 32, 32))
    assert output.shape == (1, 2, 32, 32)


def test_set_noise_model_keeps_it_out_of_state_dict(
    pn2v_noise_model: MultiChannelNoiseModel,
):
    """The noise model is attached, but not registered as a submodule."""
    module = PN2VModule(_pn2v_config().algorithm_config)

    module.set_noise_model(pn2v_noise_model)

    assert module._raw_noise_model is pn2v_noise_model
    assert not any("noise_model" in key for key in module.state_dict())
    assert "_raw_noise_model" not in dict(module.named_modules())


def test_set_noise_model_loads_npz_paths(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
):
    """Per-channel `.npz` paths are accepted, like for HDN and MicroSplit."""
    pn2v_noise_model.nmodel_0.save(str(tmp_path), "nm.npz")
    module = PN2VModule(_pn2v_config().algorithm_config)

    module.set_noise_model([tmp_path / "nm.npz"])

    assert module._raw_noise_model is not None
    assert len(module._raw_noise_model) == 1


def test_set_noise_model_rejects_wrong_channel_count():
    """A noise model needs exactly one channel per input channel."""
    module = PN2VModule(_pn2v_config().algorithm_config)

    with pytest.raises(ValueError, match="one noise model channel per input channel"):
        module.set_noise_model(gaussian_noise_model_testing(n_channels=2))


def test_fit_without_noise_model_raises(tmp_path: Path):
    """Training without a noise model fails before the first step."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config(), work_dir=tmp_path)

    with pytest.raises(ValueError, match="PN2V requires a noise model"):
        engine.train(train_data=data)


def test_pn2v_train_and_predict_smoke(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """One training step and a prediction return an MMSE image."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config("pn2v_smoke"), work_dir=tmp_path)

    engine.train(train_data=data, noise_model=pn2v_noise_model)
    predicted, _ = engine.predict(data)

    assert predicted[0].shape[0] == data.shape[0]
    assert predicted[0].shape[-2:] == data.shape[-2:]
    assert np.isfinite(predicted[0]).all()


def test_checkpoint_round_trip_restores_noise_model(
    tmp_path: Path, pn2v_noise_model: MultiChannelNoiseModel
) -> None:
    """The noise model is saved beside the weights and restored on load."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    engine = CAREamist(_pn2v_config("pn2v_ckpt"), work_dir=tmp_path)
    engine.train(train_data=data, noise_model=pn2v_noise_model)
    last = engine.get_checkpoints()[-1]

    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    loaded = PN2VModule.load_from_checkpoint(last, map_location="cpu")

    assert "noise_model" in checkpoint
    assert not any("noise_model" in key for key in checkpoint["state_dict"])
    assert loaded._raw_noise_model is not None
    torch.testing.assert_close(
        loaded._raw_noise_model.nmodel_0.weight.cpu(),
        pn2v_noise_model.nmodel_0.weight.cpu(),
    )
    for key, value in engine.model.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value.cpu())

    expected, _ = engine.predict(data)
    reloaded, _ = CAREamist(checkpoint_path=last, work_dir=tmp_path).predict(data)
    np.testing.assert_allclose(reloaded[0], expected[0], rtol=1e-5, atol=1e-4)
