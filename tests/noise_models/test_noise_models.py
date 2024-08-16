from pathlib import Path

import numpy as np
import torch
from careamics.config.nm_model import GaussianMixtureNmModel
from careamics.models.lvae.noise_models import (
    noise_model_factory,
    GaussianMixtureNoiseModel,
)


def test_gm_noise_model(tmp_path):
    nm_config = GaussianMixtureNmModel(model_type="GaussianMixtureNoiseModel")
    trained_weight = np.random.rand(18, 4)
    min_signal = np.random.rand(1)
    max_signal = np.random.rand(1)
    min_sigma = np.random.rand(1)
    filename = Path(tmp_path) / "gm_noise_model.npz"
    np.savez(
        filename,
        trained_weight=trained_weight,
        min_signal=min_signal,
        max_signal=max_signal,
        min_sigma=min_sigma,
    )
    noise_model = noise_model_factory(nm_config, [filename])
    assert noise_model is not None  # TODO wtf is nmodel_0 and why ?
    assert np.allclose(noise_model.nmodel_0.weight, trained_weight)
    assert np.allclose(noise_model.nmodel_0.min_signal, min_signal)
    assert np.allclose(noise_model.nmodel_0.max_signal, max_signal)
    assert np.allclose(noise_model.nmodel_0.min_sigma, min_sigma)
    # TODO add checks for other params, for training case


def test_gm_noise_model_training(tmp_path):
    nm_config = GaussianMixtureNmModel(model_type="GaussianMixtureNoiseModel")

    noise_model = GaussianMixtureNoiseModel(nm_config)

    # Test training
    x = np.random.rand(3)
    y = np.random.rand(3)
    output = noise_model.train(x, y, n_epochs=2)
    assert output is not None
    # TODO do something with output ?

