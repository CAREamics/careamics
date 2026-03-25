"""Tests for LVAE loss functions."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pytest
import torch

from careamics.config import (
    GaussianMixtureNMConfig,
    MultiChannelNMConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.losses.loss_factory import (
    SupportedLoss,
    loss_factory,
)
from careamics.losses.lvae.losses import (
    _compute_gaussian_log_likelihood,
    _compute_noise_model_log_likelihood,
    get_kl_divergence_loss,
    get_reconstruction_loss,
    microsplit_loss,
)
from careamics.models.lvae.noise_models import multichannel_noise_model_factory

if TYPE_CHECKING:
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel

pytestmark = pytest.mark.lvae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_dummy_noise_model_file(tmp_path, n_gaussians=3, n_coeffs=3):
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0.0]),
        "max_signal": np.array([float(2**16 - 1)]),
        "min_sigma": np.array([0.125]),
    }
    path = tmp_path / "dummy_noise_model.npz"
    np.savez(path, **nm_dict)
    return path


def init_noise_model(tmp_path, target_ch, n_gaussians=3, n_coeffs=3):
    nm_path = create_dummy_noise_model_file(tmp_path, n_gaussians, n_coeffs)
    gmm = GaussianMixtureNMConfig.from_npz(nm_path)
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    return multichannel_noise_model_factory(noise_model_config)


def _make_td_data(batch_size, n_layers, img_size, enable_lc):
    if enable_lc:
        z = [torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)][::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]
    return {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.rand(batch_size) for _ in range(n_layers)],
    }


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss_type, exp_loss_func, exp_error",
    [
        (SupportedLoss.MICROSPLIT, microsplit_loss, does_not_raise()),
        ("microsplit", microsplit_loss, does_not_raise()),
        ("made_up_loss", None, pytest.raises(NotImplementedError)),
    ],
)
def test_lvae_loss_factory(loss_type, exp_loss_func, exp_error):
    with exp_error:
        loss_func = loss_factory(loss_type)
        assert loss_func is not None
        assert callable(loss_func)
        assert loss_func == exp_loss_func


# ---------------------------------------------------------------------------
# Reconstruction loss helpers (new API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [False, True])
def test_gaussian_log_likelihood(batch_size, target_ch, predict_logvar):
    img_size = 32
    inp_ch = target_ch * (2 if predict_logvar else 1)
    reconstruction = torch.rand(batch_size, inp_ch, img_size, img_size)
    target = torch.rand(batch_size, target_ch, img_size, img_size)
    loss = _compute_gaussian_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        predict_logvar=predict_logvar,
        logvar_lowerbound=-5.0,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
def test_noise_model_log_likelihood(tmp_path, batch_size, target_ch):
    img_size = 32
    reconstruction = torch.rand(batch_size, target_ch, img_size, img_size)
    target = torch.rand(batch_size, target_ch, img_size, img_size)
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    loss = _compute_noise_model_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# KL divergence loss (unchanged API)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_lc", [False, True])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
@pytest.mark.parametrize("rescaling", ["latent_dim", "image_dim"])
@pytest.mark.parametrize("aggregation", ["mean", "sum"])
@pytest.mark.parametrize("free_bits_coeff", [0.0, 1.0])
def test_KL_divergence_loss(
    batch_size, n_layers, enable_lc, kl_type, rescaling, aggregation, free_bits_coeff
):
    img_size = 64
    if enable_lc:
        z = [torch.ones(batch_size, 128, img_size, img_size) for _ in range(n_layers)]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)][::-1]
        z = [torch.ones(batch_size, 128, sz, sz) for sz in sizes]
    td_data = {
        "z": z,
        "kl": [torch.ones(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.ones(batch_size) for _ in range(n_layers)],
    }
    kl_loss = get_kl_divergence_loss(
        kl_type=kl_type,
        topdown_data=td_data,
        rescaling=rescaling,
        aggregation=aggregation,
        free_bits_coeff=free_bits_coeff,
        img_shape=(img_size, img_size),
    )
    assert isinstance(kl_loss, torch.Tensor)
    assert isinstance(kl_loss.item(), float)


# ---------------------------------------------------------------------------
# microsplit_loss mode tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("enable_lc", [False, True])
def test_microsplit_loss_musplit_mode(batch_size, target_ch, n_layers, enable_lc):
    """Pure musplit mode: musplit_weight > 0, denoisplit_weight = 0."""
    img_size = 32
    reconstruction = torch.rand(batch_size, target_ch * 2, img_size, img_size)
    target = torch.rand(batch_size, target_ch, img_size, img_size)
    td_data = _make_td_data(batch_size, n_layers, img_size, enable_lc)
    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=1.0,
        denoisplit_weight=0.0,
        predict_logvar=True,
        logvar_lowerbound=-5.0,
    )
    output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
    )
    assert output is not None
    assert set(output.keys()) == {"loss", "reconstruction_loss", "kl_loss"}
    assert torch.isfinite(output["loss"])


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [False, True])
@pytest.mark.parametrize("n_layers", [1, 4])
def test_microsplit_loss_denoisplit_mode(tmp_path, batch_size, target_ch, predict_logvar, n_layers):
    """Pure denoisplit mode. predict_logvar=True validates the P0 chunking fix."""
    img_size = 32
    inp_ch = target_ch * (2 if predict_logvar else 1)
    reconstruction = torch.rand(batch_size, inp_ch, img_size, img_size)
    target = torch.rand(batch_size, target_ch, img_size, img_size)
    td_data = _make_td_data(batch_size, n_layers, img_size, enable_lc=False)
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=0.0,
        denoisplit_weight=1.0,
        predict_logvar=predict_logvar,
    )
    output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    assert output is not None
    assert set(output.keys()) == {"loss", "reconstruction_loss", "kl_loss"}
    assert torch.isfinite(output["loss"])


@pytest.mark.parametrize("musplit_weight", [0.1, 0.5])
def test_microsplit_loss_combined_mode(tmp_path, musplit_weight):
    """Combined musplit+denoisplit mode."""
    batch_size, target_ch, img_size = 4, 2, 32
    reconstruction = torch.rand(batch_size, target_ch * 2, img_size, img_size)
    target = torch.rand(batch_size, target_ch, img_size, img_size)
    td_data = _make_td_data(batch_size, n_layers=2, img_size=img_size, enable_lc=False)
    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=musplit_weight,
        denoisplit_weight=1.0 - musplit_weight,
        predict_logvar=True,
    )
    output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    assert output is not None
    assert set(output.keys()) == {"loss", "reconstruction_loss", "kl_loss"}
    assert torch.isfinite(output["loss"])


# ---------------------------------------------------------------------------
# Numerical equivalence tests (Tests A-E from migration plan)
# Each test inlines the legacy formula and compares numerically with the
# refactored implementation to confirm behavior preservation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("predict_logvar", [False, True])
@pytest.mark.parametrize("logvar_lowerbound", [None, -5.0])
def test_equiv_gaussian_log_likelihood(predict_logvar, logvar_lowerbound):
    """Test A: _compute_gaussian_log_likelihood matches legacy GaussianLikelihood."""
    torch.manual_seed(42)
    target_ch, img_size = 2, 16
    inp_ch = target_ch * (2 if predict_logvar else 1)
    reconstruction = torch.rand(4, inp_ch, img_size, img_size)
    target = torch.rand(4, target_ch, img_size, img_size)

    new_loss = -_compute_gaussian_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        predict_logvar=predict_logvar,
        logvar_lowerbound=logvar_lowerbound,
    )
    if predict_logvar:
        mean, logvar = reconstruction.chunk(2, dim=1)
        if logvar_lowerbound is not None:
            logvar = torch.clip(logvar, min=logvar_lowerbound)
        var = torch.exp(logvar)
        log_prob = -0.5 * (
            ((target - mean) ** 2) / var + logvar + torch.tensor(2.0 * math.pi).log()
        )
    else:
        log_prob = -0.5 * (reconstruction - target) ** 2
    legacy_loss = -log_prob.mean()

    assert torch.isclose(new_loss, legacy_loss, rtol=1e-5, atol=1e-6), (
        f"Gaussian LL mismatch: new={new_loss.item():.6f} legacy={legacy_loss.item():.6f}"
    )


def test_equiv_noise_model_log_likelihood(tmp_path):
    """Test B: _compute_noise_model_log_likelihood matches legacy NoiseModelLikelihood."""
    torch.manual_seed(42)
    target_ch, img_size = 2, 16
    nm = init_noise_model(tmp_path, target_ch)
    reconstruction = torch.rand(4, target_ch, img_size, img_size)
    target = torch.rand(4, target_ch, img_size, img_size)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    new_loss = -_compute_noise_model_log_likelihood(
        reconstruction=reconstruction,
        target=target,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    dm = torch.as_tensor(data_mean, dtype=torch.float32)
    ds = torch.as_tensor(data_std, dtype=torch.float32)
    pred_denorm = reconstruction * ds + dm
    target_denorm = target * ds + dm
    legacy_loss = -torch.log(nm.likelihood(target_denorm, pred_denorm)).mean()

    assert torch.isclose(new_loss, legacy_loss, rtol=1e-5, atol=1e-6), (
        f"NM LL mismatch: new={new_loss.item():.6f} legacy={legacy_loss.item():.6f}"
    )


@pytest.mark.parametrize("kl_weight", [0.5, 1.0])
def test_equiv_musplit_loss(kl_weight):
    """Test C: microsplit_loss musplit-mode matches inlined legacy musplit_loss."""
    torch.manual_seed(42)
    target_ch, n_layers, img_size = 2, 2, 16
    reconstruction = torch.rand(4, target_ch * 2, img_size, img_size)
    target = torch.rand(4, target_ch, img_size, img_size)
    td_data = _make_td_data(4, n_layers, img_size, enable_lc=False)

    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=1.0,
        denoisplit_weight=0.0,
        predict_logvar=True,
        logvar_lowerbound=-5.0,
        reconstruction_weight=1.0,
        kl_weight=kl_weight,
    )
    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
    )
    assert new_output is not None

    mean, logvar = reconstruction.chunk(2, dim=1)
    logvar = torch.clip(logvar, min=-5.0)
    var = torch.exp(logvar)
    log_prob = -0.5 * (
        ((target - mean) ** 2) / var + logvar + torch.tensor(2.0 * math.pi).log()
    )
    legacy_recons = -log_prob.mean()
    legacy_kl = (
        get_kl_divergence_loss(
            kl_type="kl",
            topdown_data=td_data,
            rescaling="latent_dim",
            aggregation="mean",
            free_bits_coeff=0.0,
            img_shape=(img_size, img_size),
        )
        * kl_weight
    )
    legacy_net = legacy_recons + legacy_kl

    assert torch.isclose(new_output["loss"], legacy_net, rtol=1e-4, atol=1e-5), (
        f"musplit mismatch: new={new_output['loss'].item():.6f} legacy={legacy_net.item():.6f}"
    )


def test_equiv_denoisplit_loss(tmp_path):
    """Test D: microsplit_loss denoisplit-mode matches inlined legacy denoisplit_loss.

    Also validates P0 fix: predict_logvar=True doubles channels; noise model
    receives only pred_mean (chunked from 2*C output).
    """
    torch.manual_seed(42)
    target_ch, n_layers, img_size = 2, 2, 16
    nm = init_noise_model(tmp_path, target_ch)
    reconstruction = torch.rand(4, target_ch * 2, img_size, img_size)
    target = torch.rand(4, target_ch, img_size, img_size)
    td_data = _make_td_data(4, n_layers, img_size, enable_lc=False)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=0.0,
        denoisplit_weight=1.0,
        predict_logvar=True,
        reconstruction_weight=1.0,
        kl_weight=1.0,
    )
    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    assert new_output is not None

    pred_mean, _ = reconstruction.chunk(2, dim=1)
    dm = torch.as_tensor(data_mean, dtype=torch.float32)
    ds = torch.as_tensor(data_std, dtype=torch.float32)
    legacy_recons = -torch.log(
        nm.likelihood(target * ds + dm, pred_mean * ds + dm)
    ).mean()
    legacy_kl = get_kl_divergence_loss(
        kl_type="kl_restricted",
        topdown_data=td_data,
        rescaling="image_dim",
        aggregation="sum",
        free_bits_coeff=1.0,
        img_shape=(img_size, img_size),
    )
    legacy_net = legacy_recons + legacy_kl

    assert torch.isclose(new_output["loss"], legacy_net, rtol=1e-4, atol=1e-5), (
        f"denoisplit mismatch: new={new_output['loss'].item():.6f} legacy={legacy_net.item():.6f}"
    )


def test_equiv_denoisplit_musplit_loss(tmp_path):
    """Test E: microsplit_loss combined-mode matches inlined legacy denoisplit_musplit.

    With reconstruction_weight=1.0 old and new formulas are identical.
    """
    torch.manual_seed(42)
    target_ch, n_layers, img_size = 2, 2, 16
    nm = init_noise_model(tmp_path, target_ch)
    musplit_w, denoisplit_w = 0.1, 0.9

    reconstruction = torch.rand(4, target_ch * 2, img_size, img_size)
    target = torch.rand(4, target_ch, img_size, img_size)
    td_data = _make_td_data(4, n_layers, img_size, enable_lc=False)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True) + 1e-6

    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=musplit_w,
        denoisplit_weight=denoisplit_w,
        predict_logvar=True,
        logvar_lowerbound=-5.0,
        reconstruction_weight=1.0,
        kl_weight=1.0,
    )
    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=config,
        noise_model=nm,
        data_mean=data_mean,
        data_std=data_std,
    )
    assert new_output is not None

    pred_mean, _ = reconstruction.chunk(2, dim=1)
    dm = torch.as_tensor(data_mean, dtype=torch.float32)
    ds = torch.as_tensor(data_std, dtype=torch.float32)
    nm_ll = -torch.log(nm.likelihood(target * ds + dm, pred_mean * ds + dm)).mean()
    mean, logvar = reconstruction.chunk(2, dim=1)
    logvar = torch.clip(logvar, min=-5.0)
    var = torch.exp(logvar)
    gm_ll = -(
        -0.5 * ((target - mean) ** 2 / var + logvar + torch.tensor(2.0 * math.pi).log())
    ).mean()
    legacy_recons = denoisplit_w * nm_ll + musplit_w * gm_ll

    denoisplit_kl = get_kl_divergence_loss(
        kl_type="kl_restricted",
        topdown_data=td_data,
        rescaling="image_dim",
        aggregation="sum",
        free_bits_coeff=1.0,
        img_shape=(img_size, img_size),
    )
    musplit_kl = get_kl_divergence_loss(
        kl_type="kl",
        topdown_data=td_data,
        rescaling="latent_dim",
        aggregation="mean",
        free_bits_coeff=0.0,
        img_shape=(img_size, img_size),
    )
    legacy_kl = denoisplit_w * denoisplit_kl + musplit_w * musplit_kl
    legacy_net = legacy_recons + legacy_kl

    assert torch.isclose(new_output["loss"], legacy_net, rtol=1e-4, atol=1e-5), (
        f"combined mismatch: new={new_output['loss'].item():.6f} legacy={legacy_net.item():.6f}"
    )
