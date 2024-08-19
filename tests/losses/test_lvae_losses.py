from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import GaussianLikelihoodModel, NMLikelihoodModel
from careamics.config.nm_model import GaussianMixtureNmModel, MultiChannelNmModel
from careamics.losses.loss_factory import (
    LVAELossParameters,
    SupportedLoss,
    loss_factory,
)
from careamics.losses.lvae.losses import (
    denoisplit_loss,
    get_reconstruction_loss,
    musplit_loss,
    reconstruction_loss_musplit_denoisplit,
)
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory

if TYPE_CHECKING:
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel
#     from careamics.models.lvae.likelihoods import (
#         LikelihoodModule,
#         GaussianLikelihood,
#         NoiseModelLikelihood
#     )
#     Likelihood = Union[LikelihoodModule, GaussianLikelihood, NoiseModelLikelihood]


# TODO: add this to fixtures (?)
def init_loss_parameters() -> LVAELossParameters:
    pass


# TODO: move to conftest.py as pytest.fixture
def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    """Create a dummy noise model and save it in a `.npz` file."""
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)


def init_noise_model(
    tmp_path: Path,
    target_ch: int,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> MultiChannelNoiseModel:
    """Instantiate a dummy noise model."""
    create_dummy_noise_model(tmp_path, n_gaussians, n_coeffs)
    gmm = GaussianMixtureNmModel(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNmModel(noise_models=[gmm] * target_ch)
    return noise_model_factory(noise_model_config)


@pytest.mark.parametrize(
    "loss_type, expectation",
    [
        (SupportedLoss.MUSPLIT, does_not_raise()),
        (SupportedLoss.DENOISPLIT, does_not_raise()),
        (SupportedLoss.DENOISPLIT_MUSPLIT, does_not_raise()),
        ("musplit", does_not_raise()),
        ("denoisplit", does_not_raise()),
        ("denoisplit_musplit", does_not_raise()),
        ("made_up_loss", pytest.raises(NotImplementedError)),
    ],
)
def test_lvae_loss_factory(loss_type: Union[SupportedLoss, str], expectation: Callable):
    with expectation:
        loss_func = loss_factory(loss_type)
        assert loss_func is not None
        assert callable(loss_func)


@pytest.mark.parametrize(
    "batch_size, target_ch, predict_logvar, likelihood_type",
    [
        (1, 1, None, "noise_model"),
        (1, 1, None, "gaussian"),
        (1, 1, "pixelwise", "gaussian"),
        (8, 1, None, "noise_model"),
        (8, 1, None, "gaussian"),
        (8, 1, "pixelwise", "gaussian"),
        (1, 3, None, "noise_model"),
        (1, 3, None, "gaussian"),
        (1, 3, "pixelwise", "gaussian"),
        (8, 3, None, "noise_model"),
        (8, 3, None, "gaussian"),
        (8, 3, "pixelwise", "gaussian"),
    ],
)
def test_reconstruction_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    likelihood_type: str,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    # create likelihood object
    if likelihood_type == "noise_model":
        nm = init_noise_model(tmp_path, target_ch)
        data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
        data_std = target.std(dim=(0, 2, 3), keepdim=True)
        config = NMLikelihoodModel(
            data_mean=data_mean, data_std=data_std, noise_model=nm
        )
    else:
        config = GaussianLikelihoodModel(predict_logvar=predict_logvar)
    likelihood = likelihood_factory(config)

    # compute the loss
    rec_loss = get_reconstruction_loss(
        reconstruction=reconstruction, target=target, likelihood_obj=likelihood
    )

    # check outputs
    assert rec_loss is not None
    for i in range(target_ch):
        assert rec_loss[f"ch{i+1}_loss"] is not None


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("nm_weight", [0.0, 0.5, 1.0])
def test_reconstruction_loss_musplit_denoisplit(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    nm_weight: float,
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    # create likelihood objects
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodModel(
        data_mean=data_mean, data_std=data_std, noise_model=nm
    )
    nm_likelihood = likelihood_factory(nm_config)
    gaussian_config = GaussianLikelihoodModel(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    # compute the loss
    rec_loss = reconstruction_loss_musplit_denoisplit(
        predictions=reconstruction,
        targets=target,
        nm_likelihood=nm_likelihood,
        gaussian_likelihood=gaussian_likelihood,
        nm_weight=nm_weight,
        gaussian_weight=1.0 - nm_weight,
    )

    # check outputs
    assert rec_loss is not None
    assert isinstance(rec_loss, torch.Tensor)


def test_kl_loss():
    pass


@pytest.mark.skip(reason="Implementation is likely to change soon.")
def test_musplit_loss():
    loss_func = loss_factory("musplit")
    assert loss_func == musplit_loss

    loss_parameters = LVAELossParameters
    model_outputs = torch.rand(2, 5, 64, 64)
    td_data = {
        "z": [torch.rand(2, 5, 64, 64) for _ in range(4)],
        # z is list of tensors with shape (batch, channels, height, width) for each
        # hierarchy level
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
    loss_func = loss_factory("denoisplit")
    assert loss_func == denoisplit_loss

    loss_parameters = LVAELossParameters
    model_outputs = torch.rand(2, 5, 64, 64)
    td_data = {
        "z": [torch.rand(2, 5, 64, 64) for _ in range(4)],
        # z is list of tensors with shape (batch, channels, height, width) for each
        # hierarchy level
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
