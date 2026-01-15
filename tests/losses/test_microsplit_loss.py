"""Tests for unified microsplit_loss vs legacy loss functions."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import torch

from careamics.config import (
    GaussianMixtureNMConfig,
    LVAELossConfig,
    MultiChannelNMConfig,
)
from careamics.config.losses.loss_config import KLLossConfig
from careamics.config.noise_model.likelihood_config import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.losses.lvae.losses import (
    denoisplit_loss,
    denoisplit_musplit_loss,
    microsplit_loss,
    musplit_loss,
)
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import multichannel_noise_model_factory

pytestmark = pytest.mark.lvae


def create_dummy_noise_model_file(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> Path:
    """Create a dummy noise model and save it in a `.npz` file."""
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    path = tmp_path / "dummy_noise_model.npz"
    np.savez(path, **nm_dict)
    return path


def init_noise_model(tmp_path: Path, target_ch: int):
    """Instantiate a dummy noise model."""
    nm_path = create_dummy_noise_model_file(tmp_path)
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=nm_path,
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    return multichannel_noise_model_factory(noise_model_config)


def create_test_data(
    batch_size: int,
    target_ch: int,
    img_size: int,
    n_layers: int,
    predict_logvar: str | None,
    enable_lc: bool,
    seed: int = 42,
):
    """Create test data for loss computation."""
    torch.manual_seed(seed)
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((batch_size, inp_ch, img_size, img_size))
    target = torch.rand((batch_size, target_ch, img_size, img_size))

    if enable_lc:
        z = [
            torch.rand(batch_size, 128, img_size, img_size) for _ in range(n_layers)
        ]
    else:
        sizes = [img_size // 2 ** (i + 1) for i in range(n_layers)]
        sizes = sizes[::-1]
        z = [torch.rand(batch_size, 128, sz, sz) for sz in sizes]

    td_data = {
        "z": z,
        "kl": [torch.rand(batch_size) for _ in range(n_layers)],
        "kl_restricted": [torch.rand(batch_size) for _ in range(n_layers)],
    }

    return reconstruction, target, td_data


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("target_ch", [1, 2])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [2, 4])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
def test_microsplit_matches_legacy_musplit(
    batch_size: int,
    target_ch: int,
    predict_logvar: str | None,
    n_layers: int,
    kl_type: Literal["kl", "kl_restricted"],
) -> None:
    """Verify microsplit_loss with nm_weight=0 matches musplit_loss."""
    reconstruction, target, td_data = create_test_data(
        batch_size=batch_size,
        target_ch=target_ch,
        img_size=64,
        n_layers=n_layers,
        predict_logvar=predict_logvar,
        enable_lc=True,
    )

    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    kl_params = KLLossConfig(loss_type=kl_type)
    legacy_config = LVAELossConfig(loss_type="musplit", kl_params=kl_params)

    legacy_output = musplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=legacy_config,
        gaussian_likelihood=gaussian_likelihood,
    )

    new_config = LVAELossConfig(
        loss_type="microsplit",
        kl_params=kl_params,
        musplit_weight=1.0,
        denoisplit_weight=0.0,
    )

    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=new_config,
        gaussian_likelihood=gaussian_likelihood,
        noise_model_likelihood=None,
    )

    assert legacy_output is not None
    assert new_output is not None
    torch.testing.assert_close(
        legacy_output["loss"], new_output["loss"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        legacy_output["reconstruction_loss"],
        new_output["reconstruction_loss"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        legacy_output["kl_loss"], new_output["kl_loss"], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("target_ch", [1, 2])
@pytest.mark.parametrize("n_layers", [2, 4])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
def test_microsplit_matches_legacy_denoisplit(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    n_layers: int,
    kl_type: Literal["kl", "kl_restricted"],
) -> None:
    """Verify microsplit_loss with gaussian_weight=0 matches denoisplit_loss."""
    reconstruction, target, td_data = create_test_data(
        batch_size=batch_size,
        target_ch=target_ch,
        img_size=64,
        n_layers=n_layers,
        predict_logvar=None,
        enable_lc=True,
    )

    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig()
    nm_likelihood = likelihood_factory(nm_config, noise_model=nm)
    nm_likelihood.set_data_stats(data_mean, data_std)

    kl_params = KLLossConfig(loss_type=kl_type)
    legacy_config = LVAELossConfig(loss_type="denoisplit", kl_params=kl_params)

    legacy_output = denoisplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=legacy_config,
        noise_model_likelihood=nm_likelihood,
    )

    new_config = LVAELossConfig(
        loss_type="microsplit",
        kl_params=kl_params,
        musplit_weight=0.0,
        denoisplit_weight=1.0,
    )

    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=new_config,
        gaussian_likelihood=None,
        noise_model_likelihood=nm_likelihood,
    )

    assert legacy_output is not None
    assert new_output is not None
    torch.testing.assert_close(
        legacy_output["loss"], new_output["loss"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        legacy_output["reconstruction_loss"],
        new_output["reconstruction_loss"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        legacy_output["kl_loss"], new_output["kl_loss"], rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("target_ch", [1, 2])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("n_layers", [2, 4])
@pytest.mark.parametrize("kl_type", ["kl", "kl_restricted"])
@pytest.mark.parametrize(
    "weights",
    [
        (0.1, 0.9),
        (0.5, 0.5),
        (0.9, 0.1),
    ],
)
def test_microsplit_matches_legacy_denoisplit_musplit(
    tmp_path: Path,
    batch_size: int,
    target_ch: int,
    predict_logvar: str | None,
    n_layers: int,
    kl_type: Literal["kl", "kl_restricted"],
    weights: tuple[float, float],
) -> None:
    """Verify microsplit_loss with both weights > 0 matches denoisplit_musplit_loss."""
    musplit_weight, denoisplit_weight = weights

    reconstruction, target, td_data = create_test_data(
        batch_size=batch_size,
        target_ch=target_ch,
        img_size=64,
        n_layers=n_layers,
        predict_logvar=predict_logvar,
        enable_lc=True,
    )

    nm = init_noise_model(tmp_path, target_ch)
    data_mean = target.mean(dim=(0, 2, 3), keepdim=True)
    data_std = target.std(dim=(0, 2, 3), keepdim=True)
    nm_config = NMLikelihoodConfig()
    nm_likelihood = likelihood_factory(nm_config, noise_model=nm)
    nm_likelihood.set_data_stats(data_mean, data_std)

    gaussian_config = GaussianLikelihoodConfig(predict_logvar=predict_logvar)
    gaussian_likelihood = likelihood_factory(gaussian_config)

    kl_params = KLLossConfig(loss_type=kl_type)
    legacy_config = LVAELossConfig(
        loss_type="denoisplit_musplit",
        kl_params=kl_params,
        musplit_weight=musplit_weight,
        denoisplit_weight=denoisplit_weight,
    )

    legacy_output = denoisplit_musplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=legacy_config,
        gaussian_likelihood=gaussian_likelihood,
        noise_model_likelihood=nm_likelihood,
    )

    new_config = LVAELossConfig(
        loss_type="microsplit",
        kl_params=kl_params,
        musplit_weight=musplit_weight,
        denoisplit_weight=denoisplit_weight,
    )

    new_output = microsplit_loss(
        model_outputs=(reconstruction, td_data),
        targets=target,
        config=new_config,
        gaussian_likelihood=gaussian_likelihood,
        noise_model_likelihood=nm_likelihood,
    )

    assert legacy_output is not None
    assert new_output is not None
    torch.testing.assert_close(
        legacy_output["loss"], new_output["loss"], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        legacy_output["reconstruction_loss"],
        new_output["reconstruction_loss"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        legacy_output["kl_loss"], new_output["kl_loss"], rtol=1e-5, atol=1e-5
    )


def test_microsplit_raises_without_gaussian_likelihood() -> None:
    """Test that microsplit_loss raises error when gaussian needed but missing."""
    reconstruction, target, td_data = create_test_data(
        batch_size=1,
        target_ch=2,
        img_size=64,
        n_layers=2,
        predict_logvar=None,
        enable_lc=True,
    )

    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=1.0,
        denoisplit_weight=0.0,
    )

    with pytest.raises(ValueError, match="gaussian_likelihood required"):
        microsplit_loss(
            model_outputs=(reconstruction, td_data),
            targets=target,
            config=config,
            gaussian_likelihood=None,
            noise_model_likelihood=None,
        )


def test_microsplit_raises_without_nm_likelihood() -> None:
    """Test that microsplit_loss raises error when noise model needed but missing."""
    reconstruction, target, td_data = create_test_data(
        batch_size=1,
        target_ch=2,
        img_size=64,
        n_layers=2,
        predict_logvar=None,
        enable_lc=True,
    )

    config = LVAELossConfig(
        loss_type="microsplit",
        musplit_weight=0.0,
        denoisplit_weight=1.0,
    )

    with pytest.raises(ValueError, match="noise_model_likelihood required"):
        microsplit_loss(
            model_outputs=(reconstruction, td_data),
            targets=target,
            config=config,
            gaussian_likelihood=None,
            noise_model_likelihood=None,
        )

