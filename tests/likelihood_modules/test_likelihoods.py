from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config.noise_model.noise_model_config import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)
from careamics.losses.lvae.losses import (
    _compute_gaussian_log_likelihood,
    _compute_noise_model_log_likelihood,
)
from careamics.models.lvae.noise_models import multichannel_noise_model_factory

pytestmark = pytest.mark.lvae


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [False, True])
@pytest.mark.parametrize("logvar_lowerbound", [None, 0.1])
def test_gaussian_log_likelihood(
    batch_size: int,
    target_ch: int,
    predict_logvar: bool,
    logvar_lowerbound: float | None,
) -> None:
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    log_likelihood = _compute_gaussian_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        predict_logvar=predict_logvar,
        logvar_lowerbound=logvar_lowerbound,
    )

    assert log_likelihood is not None
    assert isinstance(log_likelihood, torch.Tensor)
    assert log_likelihood.shape == torch.Size([])  # scalar


@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3, 5])
def test_noise_model_log_likelihood(
    tmp_path: Path,
    batch_size: int,
    img_size: int,
    target_ch: int,
    create_dummy_noise_model,
) -> None:
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    nm = multichannel_noise_model_factory(noise_model_config)

    inp_shape = (batch_size, target_ch, img_size, img_size)
    reconstruction = torch.rand(inp_shape)
    target = torch.rand(inp_shape)
    data_mean = target.mean().item()
    data_std = max(target.std().item(), 1e-6)

    log_likelihood = _compute_noise_model_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )

    assert log_likelihood is not None
    assert isinstance(log_likelihood, torch.Tensor)
    assert log_likelihood.shape == torch.Size([])  # scalar
