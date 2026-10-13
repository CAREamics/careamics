"""Tests for NoiseModelTrainer."""

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
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


def test_train_from_pairs_single_channel() -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 64, 64))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train_from_pairs(
        signal=signal, observation=observation, axes="SYX", n_epochs=10
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
        signal=signal, observation=observation, axes="SCYX", n_epochs=10
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
    with pytest.raises(
        ValueError,
        match="Signal and observation shapes must match after axes normalization",
    ):
        trainer.train_from_pairs(
            signal=signal,
            observation=observation,
            axes="SYX",
        )


def test_train_from_pairs_normalized_shape_mismatch_raises() -> None:
    """A channel-count mismatch is caught once both arrays are normalized."""
    signal = np.random.rand(5, 64, 64, 2)
    observation = np.random.rand(5, 64, 64, 3)

    trainer = NoiseModelTrainer()
    with pytest.raises(
        ValueError,
        match="Signal and observation shapes must match after axes normalization",
    ):
        trainer.train_from_pairs(
            signal=signal,
            observation=observation,
            axes="SYXC",
        )


def test_save_creates_files(tmp_path: Path) -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=10,
    )

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
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SYX",
        n_epochs=10,
    )
    saved_paths = trainer.save(tmp_path)

    loaded_models = NoiseModelTrainer.load(saved_paths)

    assert len(loaded_models) == 1
    assert isinstance(loaded_models[0], GaussianMixtureNoiseModel)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 32, 32))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=2, n_coeff=3)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=10,
    )
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
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=10,
    )

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
        axes="SYX",
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
        axes="SYX",
        n_epochs=500,
        learning_rate=0.1,
    )

    nm = trainer.noise_models[0]
    signal_tensor = torch.from_numpy(data["signal"]).float()
    _, sigmas, _ = nm.get_gaussian_parameters(signal_tensor)
    learned_sigma = sigmas.mean().item()

    assert np.isclose(
        learned_sigma, noise_sigma, rtol=0.15
    ), f"Learned sigma={learned_sigma:.2f}, expected sigma={noise_sigma:.2f}"


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
        axes="SYX",
        n_epochs=300,
        learning_rate=0.1,
    )

    sampled_obs = trainer.sample_observation(data["signal"], axes="SYX")

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
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=500,
    )

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
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SYX",
        n_epochs=100,
    )

    sampled = trainer.sample_observation(signal, axes="SYX")

    assert sampled.shape == signal.shape
    assert sampled.dtype == np.float64


def test_sample_observation_multi_channel() -> None:
    """Test sampling from multi-channel noise model."""
    gen = np.random.default_rng(42)
    signal = gen.uniform(0, 255, (5, 2, 64, 64))
    observation = signal + gen.normal(0, 20, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=100,
    )

    sampled = trainer.sample_observation(signal, axes="SCYX")

    assert sampled.shape == signal.shape


def test_sample_observation_without_training_raises() -> None:
    """Test that sampling without training raises error."""
    trainer = NoiseModelTrainer()
    signal = np.random.rand(5, 64, 64)

    with pytest.raises(ValueError, match="No noise models available"):
        trainer.sample_observation(signal, axes="SYX")


def test_sample_observation_distribution_matches_multichannel() -> None:
    """Test that multi-channel sampling matches true noise distribution."""
    gen = np.random.default_rng(42)
    noise_sigmas = [15.0, 35.0]

    signal = gen.uniform(0, 255, (10, 2, 64, 64))
    observation = np.empty_like(signal)
    for ch, sigma in enumerate(noise_sigmas):
        observation[:, ch] = signal[:, ch] + gen.normal(0, sigma, signal[:, ch].shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=300,
    )

    sampled = trainer.sample_observation(signal, axes="SCYX")

    for ch in range(2):
        real_noise = (observation[:, ch] - signal[:, ch]).ravel()
        synth_noise = (sampled[:, ch] - signal[:, ch]).ravel()

        distance = wasserstein_distance(real_noise / 255, synth_noise / 255)
        assert (
            distance < 0.15
        ), f"Channel {ch}: Wasserstein distance {distance:.4f} too high"


def test_trainer_global_signal_range() -> None:
    """global_signal_range=True uses global min/max across all channels."""
    gen = np.random.default_rng(1)
    signal = gen.uniform(10, 200, (4, 2, 16, 16)).astype(np.float32)
    # Ensure channels have clearly different per-channel ranges
    signal[:, 0] = gen.uniform(10, 50, (4, 16, 16))
    signal[:, 1] = gen.uniform(150, 200, (4, 16, 16))
    observation = signal + gen.normal(0, 5, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, global_signal_range=True)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )

    global_min = float(signal.min())
    global_max = float(signal.max())

    for nm in trainer.noise_models:
        assert np.isclose(
            float(nm.min_signal.item()), global_min, atol=1e-4
        ), "global min_signal mismatch"
        assert np.isclose(
            float(nm.max_signal.item()), global_max, atol=1e-4
        ), "global max_signal mismatch"


def test_get_config_returns_multichannel_nm_config() -> None:
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    gen = np.random.default_rng(2)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )

    config = trainer.get_config()

    assert isinstance(config, MultiChannelNMConfig)
    assert len(config.noise_models) == 2
    assert config.channel_indices == [0, 1]

    for ch in range(2):
        nm = trainer.noise_models[ch]
        cfg = config.noise_models[ch]
        np.testing.assert_allclose(
            np.asarray(cfg.weight),
            nm.weight.detach().cpu().numpy(),
            rtol=1e-5,
            err_msg=f"Channel {ch}: get_config() weight does not match trained model",
        )
        assert np.isclose(cfg.min_signal, float(nm.min_signal.item()))
        assert np.isclose(cfg.max_signal, float(nm.max_signal.item()))
        assert np.isclose(cfg.min_sigma, float(nm.min_sigma.item()))
        assert cfg.channel_index == ch


def test_get_config_without_training_raises() -> None:
    """get_config() raises before training."""
    trainer = NoiseModelTrainer()
    with pytest.raises(ValueError, match="No noise models available"):
        trainer.get_config()


def test_get_config_roundtrip_numerically_equivalent(tmp_path: Path) -> None:
    """get_config() weights == save-then-load-from-disk weights."""
    from careamics.config.noise_model import GaussianMixtureNMConfig as _C
    from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel as _G

    gen = np.random.default_rng(3)
    signal = gen.uniform(0, 255, (4, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 15, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SYX",
        n_epochs=5,
    )
    saved_paths = trainer.save(tmp_path)

    config_via_method = trainer.get_config()
    loaded_models = [_G(_C.from_npz(p)) for p in saved_paths]

    np.testing.assert_allclose(
        np.asarray(config_via_method.noise_models[0].weight),
        loaded_models[0].weight.detach().cpu().numpy(),
        rtol=1e-5,
        err_msg="get_config() and disk-load produce different weights",
    )


def test_config_from_paths_builds_multichannel_config(tmp_path: Path) -> None:
    """config_from_paths() builds config from saved .npz files."""
    gen = np.random.default_rng(13)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )
    saved_paths = trainer.save(tmp_path)

    config = NoiseModelTrainer.config_from_paths(saved_paths)

    assert isinstance(config, MultiChannelNMConfig)
    assert len(config.noise_models) == 2
    assert config.channel_indices == [0, 1]
    for expected_ch, cfg in enumerate(config.noise_models):
        assert cfg.channel_index == expected_ch


def test_config_from_paths_empty_paths_raises() -> None:
    """config_from_paths() raises on empty path list."""
    with pytest.raises(ValueError, match="No noise model paths provided"):
        NoiseModelTrainer.config_from_paths([])


def test_channel_indices_stored_after_training() -> None:
    gen = np.random.default_rng(4)
    signal = gen.uniform(0, 255, (4, 3, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )

    assert trainer.channel_indices == [0, 1, 2]


def test_save_embeds_channel_index_metadata(tmp_path: Path) -> None:
    gen = np.random.default_rng(5)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )
    saved_paths = trainer.save(tmp_path)

    for expected_ch, path in enumerate(saved_paths):
        data = np.load(path)
        assert "channel_index" in data, f"channel_index missing from {path.name}"
        assert int(data["channel_index"]) == expected_ch


def test_load_old_npz_without_channel_index_backward_compat(tmp_path: Path) -> None:

    weights = np.random.randn(3, 2).astype(np.float32)
    old_path = tmp_path / "old_style.npz"
    np.savez(
        old_path,
        trained_weight=weights,
        min_signal=np.array(0.0),
        max_signal=np.array(255.0),
        min_sigma=np.array(200.0),
    )

    cfg = GaussianMixtureNMConfig.from_npz(old_path)
    assert cfg.channel_index is None
    assert np.allclose(np.asarray(cfg.weight), weights)


def test_multichannel_nm_config_rejects_wrong_channel_order() -> None:
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    weights = np.ones((3, 2)).astype(np.float32)
    cfg0 = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, channel_index=0
    )
    cfg1 = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, channel_index=1
    )

    # Correct order: should succeed
    mc = MultiChannelNMConfig(noise_models=[cfg0, cfg1], channel_indices=[0, 1])
    assert mc.channel_indices == [0, 1]

    # Swapped order with metadata mismatch: should fail
    with pytest.raises(ValueError, match="Channel order mismatch"):
        MultiChannelNMConfig(noise_models=[cfg1, cfg0], channel_indices=[0, 1])


def test_multichannel_nm_config_rejects_wrong_indices_length() -> None:
    """MultiChannelNMConfig raises when channel_indices length is wrong."""
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    weights = np.ones((3, 2)).astype(np.float32)
    cfg = GaussianMixtureNMConfig(weight=weights, min_signal=0.0, max_signal=255.0)

    with pytest.raises(ValueError, match="channel_indices length"):
        MultiChannelNMConfig(noise_models=[cfg, cfg], channel_indices=[0])


def test_get_config_channel_indices_match_metadata() -> None:
    """get_config() channel_index on each GaussianMixtureNMConfig matches position."""
    gen = np.random.default_rng(6)
    signal = gen.uniform(0, 255, (4, 3, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        axes="SCYX",
        n_epochs=5,
    )

    config = trainer.get_config()
    for pos, gmm_cfg in enumerate(config.noise_models):
        assert (
            gmm_cfg.channel_index == pos
        ), f"channel_index mismatch at position {pos}: {gmm_cfg.channel_index}"


@pytest.fixture(scope="module")
def channel_last_trained() -> tuple[NoiseModelTrainer, np.ndarray, np.ndarray, list]:
    """Train on channel-last data whose two channels carry different noise."""
    gen = np.random.default_rng(42)
    sigmas = [10.0, 40.0]

    signal = gen.uniform(0, 255, (10, 64, 64, 2))
    observation = np.empty_like(signal)
    for ch, sigma in enumerate(sigmas):
        observation[..., ch] = signal[..., ch] + gen.normal(
            0, sigma, signal[..., ch].shape
        )

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=3, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=signal, observation=observation, axes="SYXC", n_epochs=500
    )
    return trainer, signal, observation, sigmas


def test_train_from_pairs_learns_per_channel_noise_from_channel_last_data(
    channel_last_trained,
) -> None:
    """Each model learns the noise of its own channel, not of a pixel row."""
    trainer, signal, _, sigmas = channel_last_trained

    assert len(trainer.noise_models) == 2
    for ch, true_sigma in enumerate(sigmas):
        signal_tensor = torch.from_numpy(signal[..., ch]).float()
        _, learned, _ = trainer.noise_models[ch].get_gaussian_parameters(signal_tensor)

        assert np.isclose(learned.mean().item(), true_sigma, rtol=0.2), (
            f"Channel {ch}: learned sigma={learned.mean().item():.2f}, "
            f"expected sigma={true_sigma:.2f}"
        )


def test_diagnose_reports_per_channel_noise_for_channel_last_data(
    channel_last_trained,
) -> None:
    """diagnose() reads each channel from the axes, not from a fixed position."""
    trainer, signal, observation, sigmas = channel_last_trained

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnostics = trainer.diagnose(
            signal=signal, observation=observation, axes="SYXC"
        )

    assert [d["channel_index"] for d in diagnostics] == [0, 1]
    for diag, true_sigma in zip(diagnostics, sigmas, strict=True):
        assert np.isclose(diag["learned_sigma_mean"], true_sigma, rtol=0.2)
        assert diag["signal_range_coverage"] > 0.99
        # a swap of the two channels lands near 0.09, a match near 0.001
        assert diag["wasserstein_distance"] < 0.02


def test_sample_observation_restores_axes_and_channel_order(
    channel_last_trained,
) -> None:
    """Samples come back in the input order, with each channel's own noise."""
    trainer, signal, _, sigmas = channel_last_trained

    sampled = trainer.sample_observation(signal, axes="SYXC")

    assert sampled.shape == signal.shape
    for ch, true_sigma in enumerate(sigmas):
        sampled_sigma = (sampled[..., ch] - signal[..., ch]).std()

        assert np.isclose(sampled_sigma, true_sigma, rtol=0.25), (
            f"Channel {ch}: sampled sigma={sampled_sigma:.2f}, "
            f"expected sigma={true_sigma:.2f}"
        )


def test_train_from_pairs_requires_axes() -> None:
    """`axes` has no default, so the layout is never guessed."""
    signal = np.random.rand(5, 2, 8, 9)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    with pytest.raises(TypeError):
        trainer.train_from_pairs(signal=signal, observation=signal)


def test_train_from_pairs_rejects_axes_shape_mismatch() -> None:
    """Axes that do not describe the array are rejected."""
    signal = np.random.rand(5, 2, 8, 9)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    with pytest.raises(ValueError, match="does not match shape"):
        trainer.train_from_pairs(
            signal=signal, observation=signal, axes="SYX", n_epochs=1
        )


def test_sample_observation_channel_count_mismatch_raises() -> None:
    """A signal whose channel count differs from the models is rejected."""
    gen = np.random.default_rng(9)
    signal = gen.uniform(0, 255, (4, 2, 8, 9))
    observation = signal + gen.normal(0, 10, signal.shape)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, min_sigma=100.0)
    trainer.train_from_pairs(
        signal=signal, observation=observation, axes="SCYX", n_epochs=5
    )

    with pytest.raises(ValueError, match="must match number of noise models"):
        trainer.sample_observation(signal[:, :1], axes="SCYX")


def test_diagnose_without_training_raises() -> None:
    """diagnose() raises before training."""
    trainer = NoiseModelTrainer()
    signal = np.random.rand(4, 2, 16, 16).astype(np.float32)
    observation = signal + 0.1

    with pytest.raises(ValueError, match="No noise models available"):
        trainer.diagnose(signal=signal, observation=observation, axes="SCYX")
