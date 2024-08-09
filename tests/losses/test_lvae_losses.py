from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import GaussianLikelihoodModel
from careamics.config.nm_model import GaussianMixtureNmModel
from careamics.losses.loss_factory import LVAELossParameters, loss_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory


def test_musplit_loss():
    loss_func = loss_factory("musplit_loss")
    assert loss_func == musplit_loss

    loss_parameters = LVAELossParameters
    model_outputs = torch.rand(2, 5, 64, 64)
    td_data = {
        "z": [
            torch.rand(2, 5, 64, 64) for _ in range(4)
        ],  # list of tensors with shape (batch, channels, height, width) for each hierarchy level
        "kl": [
            torch.rand(2) for _ in range(4)
        ],  # list of tensors with shape (batch, ) for each hierarchy level
    }
    target = torch.rand(2, 5, 64, 64)

    ll_config = GaussianLikelihoodModel(
        model_type="GaussianLikelihoodModel", color_channels=2
    )
    nm_config = GaussianMixtureNmModel(model_type="GaussianMixtureNoiseModel")

    # TODO rethink loss parameters
    loss_parameters.current_epoch = 0
    loss_parameters.inputs = torch.rand(2, 2, 5, 64, 64)
    loss_parameters.mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
    loss_parameters.likelihood = likelihood_factory(ll_config)
    loss_parameters.noise_model = noise_model_factory(nm_config)
    loss = loss_func((model_outputs, td_data), target, loss_parameters)
    for k in loss.keys():
        assert not loss[k].isnan()


@pytest.mark.skip(reason="Not implemented yet")
def test_denoisplit_loss(tmp_path):
    loss_func = loss_factory("denoisplit_loss")
    assert loss_func == denoisplit_loss

    loss_parameters = LVAELossParameters
    model_outputs = torch.rand(2, 5, 64, 64)
    td_data = {
        "z": [
            torch.rand(2, 5, 64, 64) for _ in range(4)
        ],  # list of tensors with shape (batch, channels, height, width) for each hierarchy level
        "kl": [
            torch.rand(2) for _ in range(4)
        ],  # list of tensors with shape (batch, ) for each hierarchy level
    }
    target = torch.rand(2, 5, 64, 64)

    ll_config = GaussianLikelihoodModel(
        model_type="GaussianLikelihoodModel", color_channels=2
    )
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
    )  # TODO fuckin disentwhatever class throwing forward not implemented !
    # TODO rethink loss parameters
    loss_parameters.current_epoch = 0
    loss_parameters.inputs = torch.rand(2, 2, 5, 64, 64)
    loss_parameters.mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
    loss_parameters.likelihood = likelihood_factory(ll_config)
    loss_parameters.noise_model = noise_model_factory(nm_config, [filename])
    loss = loss_func((model_outputs, td_data), target, loss_parameters)
    for k in loss.keys():
        assert not loss[k].isnan()
