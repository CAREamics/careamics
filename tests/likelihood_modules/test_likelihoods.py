from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import GaussianLikelihoodModel, NMLikelihoodModel
from careamics.config.nm_model import GaussianMixtureNmModel
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory


def test_gaussian_likelihood():
    config = GaussianLikelihoodModel(model_type="GaussianLikelihoodModel")
    likelihood = likelihood_factory(config)
    inputs = torch.rand(64, 64)
    assert likelihood(inputs, inputs)[0].mean() is not None

@pytest.mark.skip(reason="Not implemented yet")
def test_nm_likelihood(tmp_path):
    config = NMLikelihoodModel(model_type="NMLikelihoodModel")
    inputs = torch.rand(1, 1, 64, 64)

    config.data_mean = {"target": inputs.mean()}
    config.data_std = {"target": inputs.std()}

    # define noise model
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
    config.noise_model = noise_model_factory(nm_config, [filename])
    likelihood = likelihood_factory(config)
    assert likelihood(inputs, inputs)[0].mean() is not None
    # TODO add more meaningful tests
