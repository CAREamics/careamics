import torch

from careamics.config.likelihood_model import GaussianLikelihoodModel
from careamics.config.nm_model import GaussianMixtureNoiseModel
from careamics.losses.loss_factory import LVAELossParameters, loss_factory
from careamics.losses.lvae.losses import denoisplit_loss, musplit_loss
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory


def test_musplit_loss():
    loss_func = loss_factory("musplit")
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
    nm_config = GaussianMixtureNoiseModel(model_type="GaussianMixtureNoiseModel")

    # TODO rethink loss parameters
    loss_parameters.current_epoch = 0
    loss_parameters.inputs = torch.rand(2, 2, 5, 64, 64)
    loss_parameters.mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
    loss_parameters.likelihood = likelihood_factory(ll_config)
    loss_parameters.noise_model = noise_model_factory(nm_config)
    loss = loss_func((model_outputs, td_data), target, loss_parameters)
    for k in loss.keys():
        assert not loss[k].isnan()


def test_denoisplit_loss():
    loss_func = loss_factory("denoisplit")
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
    nm_config = GaussianMixtureNoiseModel(model_type="GaussianMixtureNoiseModel")

    # TODO rethink loss parameters
    loss_parameters.current_epoch = 0
    loss_parameters.inputs = torch.rand(2, 2, 5, 64, 64)
    loss_parameters.mask = ~((target == 0).reshape(len(target), -1).all(dim=1))
    loss_parameters.likelihood = likelihood_factory(ll_config)
    loss_parameters.noise_model = noise_model_factory(nm_config)
    loss = loss_func((model_outputs, td_data), target, loss_parameters)
    for k in loss.keys():
        assert not loss[k].isnan()
