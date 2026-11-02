"""Tests for transforming Gaussian mixture noise models into normalized space."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import MicroSplitLossConfig
from careamics.dataset.factory import TrainValData
from careamics.dataset.normalization.mean_std_normalization import MeanStdNormalization
from careamics.lightning.modules.microsplit_module import MicroSplitModule
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
)

pytestmark = pytest.mark.lvae

MIN_SIGNAL = 0.0
MAX_SIGNAL = 1000.0


def _make_gmm(n_gaussian=2, n_coeff=2, seed=42) -> GaussianMixtureNoiseModel:
    """Create a raw-space GMM noise model with random weights."""
    rng = np.random.default_rng(seed)
    weight = rng.random((3 * n_gaussian, n_coeff))
    config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        weight=weight,
        min_signal=MIN_SIGNAL,
        max_signal=MAX_SIGNAL,
        min_sigma=0.125,
        n_gaussian=n_gaussian,
        n_coeff=n_coeff,
    )
    return GaussianMixtureNoiseModel(config)


def _make_pairs(seed=42) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw-space signal and observation tensors within the model signal range."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(MIN_SIGNAL, MAX_SIGNAL, size=(4, 1, 16, 16))
    observation = signal + rng.normal(0.0, 1.0, size=signal.shape)
    return (
        torch.from_numpy(signal).float(),
        torch.from_numpy(observation).float(),
    )


@pytest.mark.parametrize("n_gaussian", [1, 2, 3])
@pytest.mark.parametrize("n_coeff", [2, 3])
@pytest.mark.parametrize(
    "data_mean, data_std", [(0.0, 0.1), (-100.0, 5.0), (450.0, 5000.0)]
)
def test_normalized_copy_exactness(n_gaussian, n_coeff, data_mean, data_std):
    """likelihood_norm(T(o), T(s)) must equal data_std * likelihood_raw(o, s)."""
    nm = _make_gmm(n_gaussian=n_gaussian, n_coeff=n_coeff)
    signal, observation = _make_pairs()

    nm_norm = nm.get_normalized_copy(data_mean, data_std)

    lik_raw = nm.likelihood(observation, signal)
    lik_norm = nm_norm.likelihood(
        (observation - data_mean) / data_std, (signal - data_mean) / data_std
    )

    tol = nm.tolerance
    # the additive tolerance term is outside the density, so only the density
    # part scales exactly by data_std
    torch.testing.assert_close(
        lik_norm - tol, data_std * (lik_raw - tol), rtol=1e-4, atol=1e-6
    )


def test_normalized_copy_gradient_equality():
    """Gradients w.r.t. the normalized signal must match the denormalize-then-
    evaluate path: d/ds' log lik_norm = d * d/ds log lik_raw."""
    nm = _make_gmm()
    data_mean, data_std = 450.0, 123.0
    signal, observation = _make_pairs()
    nm_norm = nm.get_normalized_copy(data_mean, data_std)

    signal_raw = signal.clone().requires_grad_()
    loss_raw = torch.log(nm.likelihood(observation, signal_raw)).sum()
    (grad_raw,) = torch.autograd.grad(loss_raw, signal_raw)

    signal_norm = ((signal - data_mean) / data_std).clone().requires_grad_()
    obs_norm = (observation - data_mean) / data_std
    loss_norm = torch.log(nm_norm.likelihood(obs_norm, signal_norm)).sum()
    (grad_norm,) = torch.autograd.grad(loss_norm, signal_norm)

    # the GMM likelihood computes in float32 internally, so the two numerical
    # paths agree only to float32 precision (a wrong scaling would be off by
    # a factor of data_std)
    torch.testing.assert_close(grad_norm, data_std * grad_raw, rtol=5e-3, atol=5e-3)


def test_normalized_copy_parameters_and_original_untouched():
    """Parameter transform is as specified and the source model is unchanged."""
    nm = _make_gmm(n_gaussian=2)
    data_mean, data_std = 100.0, 50.0
    k = nm.n_gaussian
    weight_before = nm.weight.detach().clone()

    nm_norm = nm.get_normalized_copy(data_mean, data_std)

    # original untouched
    assert not nm.is_normalized
    torch.testing.assert_close(nm.weight.detach(), weight_before)
    assert nm.min_signal.item() == MIN_SIGNAL
    assert nm.max_signal.item() == MAX_SIGNAL

    # normalized copy: mean rows scaled, variance rows shifted, alpha unchanged
    torch.testing.assert_close(
        nm_norm.weight.detach()[:k], weight_before[:k] / data_std
    )
    torch.testing.assert_close(
        nm_norm.weight.detach()[k : 2 * k],
        weight_before[k : 2 * k] - np.log(data_std**2),
    )
    torch.testing.assert_close(nm_norm.weight.detach()[2 * k :], weight_before[2 * k :])
    assert nm_norm.min_signal.item() == pytest.approx(
        (MIN_SIGNAL - data_mean) / data_std
    )
    assert nm_norm.max_signal.item() == pytest.approx(
        (MAX_SIGNAL - data_mean) / data_std
    )
    assert nm_norm.min_sigma.item() == pytest.approx(0.125 / data_std**2)
    assert nm_norm.is_normalized
    assert nm_norm.normalization_mean == data_mean
    assert nm_norm.normalization_std == data_std


def test_normalized_copy_invalid_inputs():
    """Non-positive std and double normalization are rejected."""
    nm = _make_gmm()
    with pytest.raises(ValueError, match="positive"):
        nm.get_normalized_copy(0.0, 0.0)
    nm_norm = nm.get_normalized_copy(0.0, 2.0)
    with pytest.raises(ValueError, match="already normalized"):
        nm_norm.get_normalized_copy(0.0, 2.0)


def test_save_normalized_model_raises(tmp_path):
    """The npz format stores raw-space models only."""
    nm_norm = _make_gmm().get_normalized_copy(10.0, 2.0)
    with pytest.raises(ValueError, match="normalized"):
        nm_norm.save(str(tmp_path), "nm.npz")


def test_multichannel_normalized_copy_per_channel_stats():
    """Each channel is normalized with its own statistics."""
    multi = MultiChannelNoiseModel([_make_gmm(seed=1), _make_gmm(seed=2)])
    assert not multi.is_normalized

    multi_norm = multi.get_normalized_copy([100.0, 5000.0], [10.0, 250.0])

    assert multi_norm.is_normalized
    assert multi_norm.nmodel_0.normalization_mean == 100.0
    assert multi_norm.nmodel_0.normalization_std == 10.0
    assert multi_norm.nmodel_1.normalization_mean == 5000.0
    assert multi_norm.nmodel_1.normalization_std == 250.0
    # original untouched
    assert not multi.is_normalized


def test_multichannel_normalized_copy_broadcast_and_mismatch():
    """Length-1 statistics broadcast; other length mismatches raise."""
    multi = MultiChannelNoiseModel([_make_gmm(seed=1), _make_gmm(seed=2)])

    multi_norm = multi.get_normalized_copy([100.0], [10.0])
    assert multi_norm.nmodel_0.normalization_mean == 100.0
    assert multi_norm.nmodel_1.normalization_mean == 100.0

    with pytest.raises(ValueError, match="match the number of noise models"):
        multi.get_normalized_copy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_microsplit_on_fit_start_normalizes_per_channel(tmp_path):
    """`on_fit_start` must use each channel's target statistics (not channel 0's)."""
    # raw-space noise model config with two channels
    nm = _make_gmm()
    nm.save(str(tmp_path), "nm.npz")
    gmm_config = GaussianMixtureNMConfig.from_npz(tmp_path / "nm.npz")
    nm_config = MultiChannelNMConfig(noise_models=[gmm_config] * 2)

    algorithm_config = MicroSplitAlgorithm(
        loss=MicroSplitLossConfig(
            noise_model_likelihood_weight=0.9, gaussian_likelihood_weight=0.1
        ),
        model=LVAEConfig(architecture="LVAE", output_channels=2),
    )
    module = MicroSplitModule(algorithm_config)
    module.set_noise_model(multichannel_noise_model_factory(nm_config))

    target_means = [100.0, 5000.0]
    target_stds = [10.0, 250.0]
    normalization = MeanStdNormalization(
        input_means=[0.0],
        input_stds=[1.0],
        target_means=target_means,
        target_stds=target_stds,
    )
    datamodule = SimpleNamespace(
        _data=TrainValData(
            train_data=object(),
            val_data=object(),
            train_data_target=object(),
            val_data_target=object(),
        ),
        train_dataset=SimpleNamespace(normalization=normalization),
    )
    module._trainer = SimpleNamespace(datamodule=datamodule)

    module.on_fit_start()

    assert module.noise_model is not None
    assert module.noise_model.is_normalized
    for channel_idx in range(2):
        channel_nm = getattr(module.noise_model, f"nmodel_{channel_idx}")
        assert channel_nm.normalization_mean == target_means[channel_idx]
        assert channel_nm.normalization_std == target_stds[channel_idx]

    # idempotent on repeated calls (e.g. resume): rebuilt from the raw config
    module.on_fit_start()
    assert module.noise_model.is_normalized
    assert module.noise_model.nmodel_1.normalization_mean == target_means[1]
