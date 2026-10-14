"""Tests for the PN2V likelihood head and MMSE estimate."""

from pathlib import Path

import numpy as np
import torch

from careamics import CAREamist
from careamics.config.factories import create_advanced_pn2v_config
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.lightning.modules import PN2VModule, create_module
from careamics.lightning.modules.pn2v_module import pn2v_mmse
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def _noise_model() -> GaussianMixtureNoiseModel:
    config = GaussianMixtureNMConfig(
        weight=np.zeros((3, 2), dtype=np.float32),
        n_gaussian=1,
        n_coeff=2,
        min_signal=0.0,
        max_signal=10.0,
        min_sigma=1.0,
    )
    return GaussianMixtureNoiseModel(config)


def test_likelihood_broadcasts_over_sample_channels():
    """Observations with one channel score every prior sample."""
    noise_model = _noise_model()
    observations = torch.rand(2, 1, 8, 8)
    samples = torch.rand(2, 4, 8, 8)

    likelihoods = noise_model.likelihood(observations, samples)

    assert likelihoods.shape == samples.shape


def test_pn2v_mmse_collapses_sample_axis():
    """MMSE is a weighted mean over the sample channel."""
    noise_model = _noise_model()
    observations = torch.full((1, 1, 4, 4), 1.0)
    samples = torch.rand(1, 3, 4, 4)

    estimate = pn2v_mmse(samples, observations, noise_model)

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
        noise_model=GaussianMixtureNMConfig(n_gaussian=1, n_coeff=2, min_sigma=1.0),
        num_classes=2,
        model_params={"depth": 2, "num_channels_init": 8},
    )

    module = create_module(config.algorithm_config)

    assert isinstance(module, PN2VModule)
    assert module.config.sample_scale == 10.0
    output = module(torch.rand(1, 1, 32, 32))
    assert output.shape == (1, 2, 32, 32)


def test_pn2v_train_and_predict_smoke(tmp_path: Path) -> None:
    """One training step and a prediction return an MMSE image."""
    data = np.random.default_rng(0).random((2, 32, 32), dtype=np.float32) * 100
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_smoke",
        data_type="array",
        axes="SYX",
        patch_size=(16, 16),
        batch_size=2,
        noise_model=GaussianMixtureNMConfig(
            n_gaussian=1,
            n_coeff=2,
            min_signal=0.0,
            max_signal=100.0,
            min_sigma=1.0,
        ),
        num_epochs=1,
        num_steps=1,
        num_classes=2,
        model_params={"depth": 2, "num_channels_init": 8, "use_batch_norm": False},
        n_val_patches=1,
        masked_pixel_percentage=5,
        roi_size=5,
    )

    engine = CAREamist(config, work_dir=tmp_path)
    engine.train(train_data=data)
    predicted, _ = engine.predict(data)

    assert predicted[0].shape[0] == data.shape[0]
    assert predicted[0].shape[-2:] == data.shape[-2:]
    assert np.isfinite(predicted[0]).all()
