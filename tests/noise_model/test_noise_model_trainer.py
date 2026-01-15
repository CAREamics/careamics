"""Tests for NoiseModelTrainer."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)
from careamics.noise_model import NoiseModelTrainer


@pytest.fixture
def synthetic_noisy_data() -> Callable:
    """Factory fixture for creating synthetic signal-observation pairs."""

    def _create(
        shape: tuple[int, ...],
        noise_sigma: float,
        signal_range: tuple[float, float] = (0.0, 255.0),
        seed: int = 42,
    ) -> dict:
        gen = np.random.default_rng(seed)

        signal = gen.uniform(signal_range[0], signal_range[1], shape)
        noise = gen.normal(0, noise_sigma, shape)
        observation = signal + noise

        return {
            "signal": signal,
            "observation": observation,
            "noise_sigma": noise_sigma,
            "signal_range": signal_range,
        }

    return _create


def test_init_default_params() -> None:
    trainer = NoiseModelTrainer()
    assert trainer.n_gaussian == 3
    assert trainer.n_coeff == 3
    assert trainer.min_sigma == 125.0
    assert trainer.noise_models is None
    assert trainer.histograms is None


def test_init_custom_params() -> None:
    trainer = NoiseModelTrainer(n_gaussian=5, n_coeff=4, min_sigma=100.0)
    assert trainer.n_gaussian == 5
    assert trainer.n_coeff == 4
    assert trainer.min_sigma == 100.0


@pytest.mark.parametrize(
    "axes,shape,expected",
    [
        ("SYX", (5, 64, 64), 1),
        ("SCYX", (5, 2, 64, 64), 2),
        ("SCYX", (5, 3, 64, 64), 3),
        ("SCZYX", (5, 4, 8, 64, 64), 4),
        ("YX", (64, 64), 1),
    ],
)
def test_get_n_channels(
    axes: str, shape: tuple[int, ...], expected: int
) -> None:
    data = np.zeros(shape)
    result = NoiseModelTrainer._get_n_channels(data, axes)
    assert result == expected


@pytest.mark.parametrize(
    "axes,shape,channel_idx,expected_shape",
    [
        ("SCYX", (5, 2, 64, 64), 0, (5, 64, 64)),
        ("SCYX", (5, 2, 64, 64), 1, (5, 64, 64)),
        ("SCYX", (5, 3, 32, 32), 2, (5, 32, 32)),
        ("SCZYX", (2, 2, 4, 16, 16), 0, (2, 4, 16, 16)),
        ("SYX", (5, 64, 64), 0, (5, 64, 64)),
    ],
)
def test_extract_channel(
    axes: str,
    shape: tuple[int, ...],
    channel_idx: int,
    expected_shape: tuple[int, ...],
) -> None:
    data = np.random.rand(*shape)
    result = NoiseModelTrainer._extract_channel(data, channel_idx, axes)
    assert result.shape == expected_shape


def test_extract_channel_values() -> None:
    data = np.arange(2 * 2 * 4 * 4).reshape((2, 2, 4, 4))
    channel_0 = NoiseModelTrainer._extract_channel(data, 0, "SCYX")
    channel_1 = NoiseModelTrainer._extract_channel(data, 1, "SCYX")
    np.testing.assert_array_equal(channel_0, data[:, 0, :, :])
    np.testing.assert_array_equal(channel_1, data[:, 1, :, :])


@pytest.mark.parametrize(
    "axes,expected",
    [
        ("SCYX", "SYX"),
        ("SCZYX", "SZYX"),
        ("CYX", "YX"),
        ("SYX", "SYX"),
    ],
)
def test_remove_channel_axis(axes: str, expected: str) -> None:
    result = NoiseModelTrainer._remove_channel_axis(axes)
    assert result == expected


def test_train_from_pairs_single_channel() -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 64, 64))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train_from_pairs(
        signal=signal, observation=observation, n_epochs=10
    )

    assert len(noise_models) == 1
    assert isinstance(noise_models[0], GaussianMixtureNoiseModel)
    assert trainer.noise_models is not None
    assert trainer.histograms is not None
    assert len(trainer.histograms) == 1


def test_train_from_pairs_multi_channel() -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 64, 64))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train_from_pairs(
        signal=signal, observation=observation, n_epochs=10
    )

    assert len(noise_models) == 2
    assert all(isinstance(nm, GaussianMixtureNoiseModel) for nm in noise_models)
    assert trainer.noise_models is not None
    assert len(trainer.noise_models) == 2
    assert trainer.histograms is not None
    assert len(trainer.histograms) == 2


def test_train_from_pairs_shape_mismatch_raises() -> None:
    signal = np.random.rand(5, 64, 64)
    observation = np.random.rand(5, 32, 32)

    trainer = NoiseModelTrainer()
    with pytest.raises(ValueError, match="Signal and observation shapes must match"):
        trainer.train_from_pairs(signal=signal, observation=observation)


def test_train_with_clean_data_single_channel() -> None:
    gen = np.random.default_rng(42)
    noisy_data = gen.uniform(0, 255, (5, 64, 64))
    clean_data = noisy_data - gen.normal(0, 10, noisy_data.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train(
        noisy_data=noisy_data,
        axes="SYX",
        patch_size=[32, 32],
        clean_data=clean_data,
        nm_epochs=10,
    )

    assert len(noise_models) == 1
    assert isinstance(noise_models[0], GaussianMixtureNoiseModel)
    assert trainer._n2v_model is None


def test_train_with_clean_data_multi_channel() -> None:
    gen = np.random.default_rng(42)
    noisy_data = gen.uniform(0, 255, (5, 2, 64, 64))
    clean_data = noisy_data - gen.normal(0, 10, noisy_data.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train(
        noisy_data=noisy_data,
        axes="SCYX",
        patch_size=[32, 32],
        clean_data=clean_data,
        nm_epochs=10,
    )

    assert len(noise_models) == 2
    assert all(isinstance(nm, GaussianMixtureNoiseModel) for nm in noise_models)
    assert trainer._n2v_model is None


def test_train_clean_data_shape_mismatch_raises() -> None:
    noisy_data = np.random.rand(5, 64, 64)
    clean_data = np.random.rand(5, 32, 32)

    trainer = NoiseModelTrainer()
    with pytest.raises(ValueError, match="clean_data shape must match noisy_data"):
        trainer.train(
            noisy_data=noisy_data,
            axes="SYX",
            patch_size=[32, 32],
            clean_data=clean_data,
        )


def test_save_creates_files(tmp_path: Path) -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=10)

    saved_paths = trainer.save(tmp_path, prefix="test_nm")

    assert len(saved_paths) == 2
    assert all(p.exists() for p in saved_paths)
    assert saved_paths[0].name == "test_nm_ch0.npz"
    assert saved_paths[1].name == "test_nm_ch1.npz"


def test_save_without_training_raises(tmp_path: Path) -> None:
    trainer = NoiseModelTrainer()
    with pytest.raises(ValueError, match="No noise models to save"):
        trainer.save(tmp_path)


def test_load_models(tmp_path: Path) -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=10)
    saved_paths = trainer.save(tmp_path)

    loaded_models = NoiseModelTrainer.load(saved_paths)

    assert len(loaded_models) == 1
    assert isinstance(loaded_models[0], GaussianMixtureNoiseModel)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=2, n_coeff=3)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=10)
    saved_paths = trainer.save(tmp_path)

    loaded_models = NoiseModelTrainer.load(saved_paths)

    assert len(loaded_models) == 2
    for orig, loaded in zip(trainer.noise_models, loaded_models, strict=True):
        np.testing.assert_array_almost_equal(
            orig.weight.cpu().numpy(),
            loaded.weight.cpu().numpy(),
        )
        assert orig.n_gaussian == loaded.n_gaussian
        assert orig.n_coeff == loaded.n_coeff


def test_get_multichannel_model() -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=10)

    multichannel = trainer.get_multichannel_model()

    assert isinstance(multichannel, MultiChannelNoiseModel)
    assert multichannel._nm_cnt == 2


def test_get_multichannel_model_without_training_raises() -> None:
    trainer = NoiseModelTrainer()
    with pytest.raises(ValueError, match="No noise models available"):
        trainer.get_multichannel_model()


def test_trained_model_has_weights() -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (10, 64, 64))
    noise = gen.normal(0, 25, signal.shape)
    observation = signal + noise

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        n_epochs=50,
        learning_rate=0.1,
    )

    nm = trainer.noise_models[0]
    assert nm is not None
    assert nm.weight is not None


@pytest.mark.parametrize("noise_sigma", [10.0, 25.0, 50.0])
def test_noise_model_learns_correct_sigma(
    synthetic_noisy_data: Callable, noise_sigma: float
) -> None:
    """Verify noise model learns the true noise standard deviation."""
    data = synthetic_noisy_data(
        shape=(10, 64, 64),
        noise_sigma=noise_sigma,
        signal_range=(0.0, 255.0),
    )

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=data["signal"],
        observation=data["observation"],
        n_epochs=500,
        learning_rate=0.1,
    )

    nm = trainer.noise_models[0]
    signal_tensor = torch.from_numpy(data["signal"]).float()
    _, sigmas, _ = nm.get_gaussian_parameters(signal_tensor)
    learned_sigma = sigmas.mean().item()

    assert np.isclose(learned_sigma, noise_sigma, rtol=0.15), (
        f"Learned sigma={learned_sigma:.2f}, expected sigma={noise_sigma:.2f}"
    )


@pytest.mark.parametrize("noise_sigma", [15.0, 30.0])
def test_noise_model_samples_match_distribution(
    synthetic_noisy_data: Callable, noise_sigma: float
) -> None:
    """Verify sampled noise matches true noise distribution."""
    data = synthetic_noisy_data(
        shape=(10, 128, 128),
        noise_sigma=noise_sigma,
        signal_range=(0.0, 255.0),
    )

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=data["signal"],
        observation=data["observation"],
        n_epochs=300,
        learning_rate=0.1,
    )

    sampled_obs = trainer.sample_observation(data["signal"])

    real_noise = (data["observation"] - data["signal"]).ravel()
    sampled_noise = (sampled_obs - data["signal"]).ravel()

    scale = data["signal_range"][1]
    distance = wasserstein_distance(real_noise / scale, sampled_noise / scale)

    assert distance < 0.1, f"Wasserstein distance {distance:.4f} too high"


def test_multichannel_learns_different_noise_levels() -> None:
    """Verify multi-channel model learns different sigma per channel."""
    gen = np.random.default_rng(42)
    noise_sigmas = [10.0, 40.0]

    signal = gen.uniform(0, 255, (10, 2, 64, 64))
    observation = np.empty_like(signal)
    for ch, sigma in enumerate(noise_sigmas):
        observation[:, ch] = signal[:, ch] + gen.normal(0, sigma, signal[:, ch].shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=500)

    for ch, true_sigma in enumerate(noise_sigmas):
        nm = trainer.noise_models[ch]
        signal_tensor = torch.from_numpy(signal[:, ch]).float()
        _, sigmas, _ = nm.get_gaussian_parameters(signal_tensor)
        learned_sigma = sigmas.mean().item()

        assert np.isclose(learned_sigma, true_sigma, rtol=0.2), (
            f"Channel {ch}: learned sigma={learned_sigma:.2f}, "
            f"expected sigma={true_sigma:.2f}"
        )


def test_sample_observation_single_channel() -> None:
    """Test sampling from single-channel noise model."""
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 64, 64))
    observation = signal + gen.normal(0, 20, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=100)

    sampled = trainer.sample_observation(signal)

    assert sampled.shape == signal.shape
    assert sampled.dtype == np.float64


def test_sample_observation_multi_channel() -> None:
    """Test sampling from multi-channel noise model."""
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 64, 64))
    observation = signal + gen.normal(0, 20, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=100)

    sampled = trainer.sample_observation(signal)

    assert sampled.shape == signal.shape


def test_sample_observation_without_training_raises() -> None:
    """Test that sampling without training raises error."""
    trainer = NoiseModelTrainer()
    signal = np.random.rand(5, 64, 64)

    with pytest.raises(ValueError, match="No noise models available"):
        trainer.sample_observation(signal)


def test_sample_observation_distribution_matches_multichannel() -> None:
    """Test that multi-channel sampling matches true noise distribution."""
    gen = np.random.default_rng(42)
    noise_sigmas = [15.0, 35.0]

    signal = gen.uniform(0, 255, (10, 2, 64, 64))
    observation = np.empty_like(signal)
    for ch, sigma in enumerate(noise_sigmas):
        observation[:, ch] = signal[:, ch] + gen.normal(0, sigma, signal[:, ch].shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=300)

    sampled = trainer.sample_observation(signal)

    for ch in range(2):
        real_noise = (observation[:, ch] - signal[:, ch]).ravel()
        synth_noise = (sampled[:, ch] - signal[:, ch]).ravel()

        distance = wasserstein_distance(real_noise / 255, synth_noise / 255)
        assert distance < 0.15, (
            f"Channel {ch}: Wasserstein distance {distance:.4f} too high"
        )

