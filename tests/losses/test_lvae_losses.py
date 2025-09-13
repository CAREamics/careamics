from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pytest
import torch

from careamics.config import (
    GaussianMixtureNMConfig,
    LVAELossConfig,
    MultiChannelNMConfig,
)
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import KLLossConfig
from careamics.losses.loss_factory import (
    SupportedLoss,
    loss_factory,
)
from careamics.losses.lvae.losses import (
    _reconstruction_loss_musplit_denoisplit,
    denoisplit_loss,
    denoisplit_musplit_loss,
    get_kl_divergence_loss,
    get_reconstruction_loss,
    musplit_loss,
)
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory

if TYPE_CHECKING:
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel

pytestmark = pytest.mark.lvae


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
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    return noise_model_factory(noise_model_config)


@pytest.mark.parametrize(
    "loss_type, exp_loss_func, exp_error",
    [
        (SupportedLoss.MUSPLIT, musplit_loss, does_not_raise()),
        (SupportedLoss.DENOISPLIT, denoisplit_loss, does_not_raise()),
        (SupportedLoss.DENOISPLIT_MUSPLIT, denoisplit_musplit_loss, does_not_raise()),
        ("musplit", musplit_loss, does_not_raise()),
        ("denoisplit", denoisplit_loss, does_not_raise()),
        ("denoisplit_musplit", denoisplit_musplit_loss, does_not_raise()),
        ("made_up_loss", None, pytest.raises(NotImplementedError)),
    ],
)
def test_lvae_loss_factory(
    loss_type: Union[SupportedLoss, str], exp_loss_func: Callable, exp_error: Callable
):
    with exp_error:
        loss_func = loss_factory(loss_type)
        assert loss_func is not None
        assert callable(loss_func)
        assert loss_func == exp_loss_func


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
        config = NMLikelihoodConfig()
        likelihood = likelihood_factory(config, noise_model=nm)
        likelihood.set_data_stats(data_mean, data_std)
    else:
        nm = None
        config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
        likelihood = likelihood_factory(config, noise_model=nm)

    # compute the loss
    rec_loss = get_reconstruction_loss(
        reconstruction=reconstruction, target=target, likelihood_obj=likelihood
    )

    # check outputs
    assert rec_loss is not None


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
    nm_config = NMLikelihoodConfig()
    nm_likelihood = likelihood_factory(nm_config, noise_model=nm)
    nm_likelihood.set_data_stats(data_mean, data_std)
    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    # compute the loss
    rec_loss = _reconstruction_loss_musplit_denoisplit(
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


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_LC", [False, True])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
@pytest.mark.parametrize("rescaling", ["latent_dim", "image_dim"])
@pytest.mark.parametrize("aggregation", ["mean", "sum"])
@pytest.mark.parametrize("free_bits_coeff", [0.0, 1.0])
def test_KL_divergence_loss(
    batch_size: int,
    n_layers: int,
    enable_LC: bool,
    kl_type: Literal["kl", "kl_restricted"],
    rescaling: Literal["latent_dim", "image_dim"],
    aggregation: Literal["mean", "sum"],
    free_bits_coeff: float,
):
    # create test data
    img_size = 64
    if enable_LC:
        z = [torch.ones(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.ones(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.ones(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.ones(batch_size) for _ in range(n_layers)],
    }

    # compute the loss for different settings
    img_shape = (img_size, img_size)
    kl_loss = get_kl_divergence_loss(
        kl_type=kl_type,
        topdown_data=td_data,
        rescaling=rescaling,
        aggregation=aggregation,
        free_bits_coeff=free_bits_coeff,
        img_shape=img_shape,
    )
    assert isinstance(kl_loss, torch.Tensor)
    assert isinstance(kl_loss.item(), float)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_LC", [False, True])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
def test_musplit_loss(
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    n_layers: int,
    enable_LC: bool,
    kl_type: Literal["kl", "kl_restricted"],
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    if enable_LC:
        z = [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood
    config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    likelihood = likelihood_factory(config)

    # compute the loss
    kl_params = KLLossConfig(loss_type=kl_type)
    loss_parameters = LVAELossConfig(loss_type="musplit", kl_params=kl_params)
    output = musplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=loss_parameters,
        gaussian_likelihood=likelihood,
    )

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
def test_denoisplit_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    n_layers: int,
    kl_type: Literal["kl", "kl_restricted"],
):
    # create test data
    img_size = 64
    reconstruction = torch.rand((batch_size, target_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    td_data = {
        "z": [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)],
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig()
    likelihood = likelihood_factory(nm_config, noise_model=nm)
    likelihood.set_data_stats(data_mean, data_std)

    # compute the loss
    kl_params = KLLossConfig(loss_type=kl_type)
    loss_parameters = LVAELossConfig(loss_type="denoisplit", kl_params=kl_params)
    output = denoisplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=loss_parameters,
        noise_model_likelihood=likelihood,
    )

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_LC", [False, True])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
def test_denoisplit_musplit_loss(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str,
    n_layers: int,
    enable_LC: bool,
    kl_type: Literal["kl", "kl_restricted"],
):
    # create test data
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))
    if enable_LC:
        z = [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    # create likelihood objects
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig()
    nm_likelihood = likelihood_factory(nm_config, noise_model=nm)
    nm_likelihood.set_data_stats(data_mean, data_std)
    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    # compute the loss
    kl_params = KLLossConfig(loss_type=kl_type)
    loss_parameters = LVAELossConfig(
        loss_type="denoisplit_musplit", kl_params=kl_params
    )
    output = denoisplit_musplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=loss_parameters,
        gaussian_likelihood=gaussian_likelihood,
        noise_model_likelihood=nm_likelihood,
    )

    # check outputs
    # NOTE: output should not be None in these test cases
    assert output is not None
    assert isinstance(output, dict)
    assert "loss" in output
    assert "reconstruction_loss" in output
    assert "kl_loss" in output
