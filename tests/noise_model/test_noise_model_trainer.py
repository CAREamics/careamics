"""Tests for NoiseModelTrainer."""

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


def test_train_from_pairs_accepts_separate_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = np.random.default_rng(42)
    signal_scyx = gen.uniform(0, 255, (3, 2, 8, 9))
    observation_scyx = signal_scyx + gen.normal(0, 10, signal_scyx.shape)
    observation_syxc = np.moveaxis(observation_scyx, 1, -1)
    captured_channels: list[tuple[np.ndarray, np.ndarray]] = []

    def _fake_train_single_channel(
        self: NoiseModelTrainer,
        signal: np.ndarray,
        observation: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        min_signal: float | None = None,
        max_signal: float | None = None,
    ) -> tuple[GaussianMixtureNoiseModel, list[float]]:
        _ = (n_epochs, learning_rate, batch_size)
        captured_channels.append((signal.copy(), observation.copy()))
        config = GaussianMixtureNMConfig(
            min_signal=min_signal if min_signal is not None else float(signal.min()),
            max_signal=max_signal if max_signal is not None else float(signal.max()),
            n_gaussian=self.n_gaussian,
            n_coeff=self.n_coeff,
            min_sigma=self.min_sigma,
        )
        return GaussianMixtureNoiseModel(config), [0.0]

    monkeypatch.setattr(
        NoiseModelTrainer,
        "_train_single_channel",
        _fake_train_single_channel,
    )

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    noise_models = trainer.train_from_pairs(
        signal=signal_scyx,
        observation=observation_syxc,
        signal_axes="SCYX",
        observation_axes="SYXC",
        n_epochs=1,
    )

    assert len(noise_models) == 2
    assert trainer.channel_indices == [0, 1]
    assert len(captured_channels) == 2
    np.testing.assert_array_equal(captured_channels[0][0], signal_scyx[:, 0])
    np.testing.assert_array_equal(captured_channels[0][1], observation_scyx[:, 0])
    np.testing.assert_array_equal(captured_channels[1][0], signal_scyx[:, 1])
    np.testing.assert_array_equal(captured_channels[1][1], observation_scyx[:, 1])


def test_train_from_pairs_shape_mismatch_raises() -> None:
    signal = np.random.rand(5, 64, 64)
    observation = np.random.rand(5, 32, 32)

    trainer = NoiseModelTrainer()
    with pytest.raises(
        ValueError,
        match="Signal and observation shapes must match after axes normalization",
    ):
        trainer.train_from_pairs(signal=signal, observation=observation)


def test_train_from_pairs_normalized_shape_mismatch_raises() -> None:
    signal = np.random.rand(5, 2, 64, 64)
    observation = np.random.rand(5, 64, 64, 3)

    trainer = NoiseModelTrainer()
    with pytest.raises(
        ValueError,
        match="Signal and observation shapes must match after axes normalization",
    ):
        trainer.train_from_pairs(
            signal=signal,
            observation=observation,
            signal_axes="SCYX",
            observation_axes="SYXC",
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
        assert (
            distance < 0.15
        ), f"Channel {ch}: Wasserstein distance {distance:.4f} too high"


# ---------------------------------------------------------------------------
# Equivalence test: NoiseModelTrainer must produce numerically identical
# results to the reference MicroSplit training loop when configured to use
# the same per-channel min/max signal range (trainer default behaviour).
# ---------------------------------------------------------------------------


def test_trainer_equivalent_to_reference_loop_per_channel_range() -> None:
    """NoiseModelTrainer matches reference loop with per-channel signal range.

    The reference loop is re-implemented here verbatim except that channel
    extraction uses ``[:, channel_idx]`` (SCYX order, matching the trainer)
    rather than ``[..., channel_idx]`` which would be channel-last.

    Both paths are seeded with the same torch seed so that weight
    initialisation and shuffle order are identical.
    """
    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel

    gen = np.random.default_rng(0)
    n_samples, n_channels, height, width = 4, 2, 16, 16
    signal = gen.uniform(0, 255, (n_samples, n_channels, height, width)).astype(
        np.float32
    )
    observation = (signal + gen.normal(0, 20, signal.shape)).astype(np.float32)

    n_gaussian, n_coeff, min_sigma = 2, 3, 125.0
    n_epochs, lr, batch_size = 5, 0.1, 250000

    # --- reference loop (per-channel min/max, matching trainer default) ---
    ref_models = []
    for ch in range(n_channels):
        ch_signal = signal[:, ch]  # (S, H, W)
        ch_obs = observation[:, ch]
        cfg = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            min_signal=float(ch_signal.min()),
            max_signal=float(ch_signal.max()),
            n_gaussian=n_gaussian,
            n_coeff=n_coeff,
            min_sigma=min_sigma,
        )
        torch.manual_seed(42 + ch)
        nm = GaussianMixtureNoiseModel(cfg)
        ch_signal_flat = ch_signal.reshape(-1, *ch_signal.shape[-2:])
        ch_obs_flat = ch_obs.reshape(-1, *ch_obs.shape[-2:])
        torch.manual_seed(42 + ch)
        nm.fit(
            signal=ch_signal_flat,
            observation=ch_obs_flat,
            learning_rate=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        ref_models.append(nm)

    # --- trainer path (per-channel, default) ---
    trainer = NoiseModelTrainer(
        n_gaussian=n_gaussian, n_coeff=n_coeff, min_sigma=min_sigma
    )

    # Patch _train_single_channel to inject deterministic seeds
    original_method = trainer._train_single_channel.__func__

    ch_call_count = [0]

    def seeded_train(self, signal, observation, n_epochs, learning_rate,
                     batch_size, min_signal=None, max_signal=None):
        ch = ch_call_count[0]
        torch.manual_seed(42 + ch)
        ch_call_count[0] += 1
        # rebuild with same seed for init
        from careamics.config.noise_model import GaussianMixtureNMConfig as _C
        from careamics.models.lvae.noise_models import (
            GaussianMixtureNoiseModel as _G,
        )
        cfg = _C(
            model_type="GaussianMixtureNoiseModel",
            min_signal=min_signal if min_signal is not None else float(signal.min()),
            max_signal=max_signal if max_signal is not None else float(signal.max()),
            n_gaussian=self.n_gaussian,
            n_coeff=self.n_coeff,
            min_sigma=self.min_sigma,
        )
        nm = _G(cfg)
        sig_flat = signal.reshape(-1, *signal.shape[-2:])
        obs_flat = observation.reshape(-1, *observation.shape[-2:])
        torch.manual_seed(42 + ch)
        losses = nm.fit(
            signal=sig_flat,
            observation=obs_flat,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )
        return nm, losses

    import types
    trainer._train_single_channel = types.MethodType(seeded_train, trainer)

    trainer.train_from_pairs(
        signal=signal,
        observation=observation,
        n_epochs=n_epochs,
        learning_rate=lr,
        batch_size=batch_size,
    )

    for ch in range(n_channels):
        np.testing.assert_allclose(
            trainer.noise_models[ch].weight.detach().cpu().numpy(),
            ref_models[ch].weight.detach().cpu().numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Channel {ch}: trainer weight differs from reference loop",
        )
        assert np.isclose(
            float(trainer.noise_models[ch].min_signal.item()),
            float(ref_models[ch].min_signal.item()),
        )
        assert np.isclose(
            float(trainer.noise_models[ch].max_signal.item()),
            float(ref_models[ch].max_signal.item()),
        )


def test_trainer_global_signal_range() -> None:
    """global_signal_range=True uses global min/max across all channels."""
    gen = np.random.default_rng(1)
    signal = gen.uniform(10, 200, (4, 2, 16, 16)).astype(np.float32)
    # Ensure channels have clearly different per-channel ranges
    signal[:, 0] = gen.uniform(10, 50, (4, 16, 16))
    signal[:, 1] = gen.uniform(150, 200, (4, 16, 16))
    observation = signal + gen.normal(0, 5, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2, global_signal_range=True)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)

    global_min = float(signal.min())
    global_max = float(signal.max())

    for nm in trainer.noise_models:
        assert np.isclose(float(nm.min_signal.item()), global_min, atol=1e-4), (
            "global min_signal mismatch"
        )
        assert np.isclose(float(nm.max_signal.item()), global_max, atol=1e-4), (
            "global max_signal mismatch"
        )


# ---------------------------------------------------------------------------
# Issue #850: NoiseModelTrainer.get_config() returns MultiChannelNMConfig
# ---------------------------------------------------------------------------


def test_get_config_returns_multichannel_nm_config() -> None:
    """Issue #850: get_config() returns a MultiChannelNMConfig without disk I/O."""
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    gen = np.random.default_rng(2)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)

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
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)
    saved_paths = trainer.save(tmp_path)

    config_via_method = trainer.get_config()
    loaded_models = [_G(_C.from_npz(p)) for p in saved_paths]

    np.testing.assert_allclose(
        np.asarray(config_via_method.noise_models[0].weight),
        loaded_models[0].weight.detach().cpu().numpy(),
        rtol=1e-5,
        err_msg="get_config() and disk-load produce different weights",
    )


# ---------------------------------------------------------------------------
# Issue #851: Channel order metadata and validation
# ---------------------------------------------------------------------------


def test_channel_indices_stored_after_training() -> None:
    """Issue #851: channel_indices are stored and match training order."""
    gen = np.random.default_rng(4)
    signal = gen.uniform(0, 255, (4, 3, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)

    assert trainer.channel_indices == [0, 1, 2]


def test_save_embeds_channel_index_metadata(tmp_path: Path) -> None:
    """Issue #851: saved .npz files include channel_index."""
    gen = np.random.default_rng(5)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)
    saved_paths = trainer.save(tmp_path)

    for expected_ch, path in enumerate(saved_paths):
        data = np.load(path)
        assert "channel_index" in data, f"channel_index missing from {path.name}"
        assert int(data["channel_index"]) == expected_ch


def test_load_old_npz_without_channel_index_backward_compat(tmp_path: Path) -> None:
    """Issue #851: .npz files without channel_index still load correctly."""
    from careamics.config.noise_model import GaussianMixtureNMConfig

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
    """Issue #851: MultiChannelNMConfig raises on channel_index mismatch."""
    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    weights = np.ones((3, 2)).astype(np.float32)
    cfg0 = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, channel_index=0
    )
    cfg1 = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, channel_index=1
    )

    # Correct order: should succeed
    mc = MultiChannelNMConfig(
        noise_models=[cfg0, cfg1], channel_indices=[0, 1]
    )
    assert mc.channel_indices == [0, 1]

    # Swapped order with metadata mismatch: should fail
    with pytest.raises(ValueError, match="Channel order mismatch"):
        MultiChannelNMConfig(noise_models=[cfg1, cfg0], channel_indices=[0, 1])


def test_multichannel_nm_config_rejects_wrong_indices_length() -> None:
    """MultiChannelNMConfig raises when channel_indices length is wrong."""
    from careamics.config.noise_model import GaussianMixtureNMConfig
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
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)

    config = trainer.get_config()
    for pos, gmm_cfg in enumerate(config.noise_models):
        assert gmm_cfg.channel_index == pos, (
            f"channel_index mismatch at position {pos}: {gmm_cfg.channel_index}"
        )


# ---------------------------------------------------------------------------
# Issue #849: nm_paths removed from create_microsplit_configuration
# ---------------------------------------------------------------------------


def test_create_microsplit_configuration_no_nm_paths_param() -> None:
    """Issue #849: create_microsplit_configuration does not accept nm_paths."""
    import inspect

    from careamics.config.configuration_factories import (
        create_microsplit_configuration,
    )

    sig = inspect.signature(create_microsplit_configuration)
    assert "nm_paths" not in sig.parameters, (
        "nm_paths should have been removed from create_microsplit_configuration "
        "(issue #849)"
    )


def test_create_microsplit_configuration_without_noise_model_prints_reminder(
    capsys,
) -> None:
    """Issue #849: factory prints a reminder when denoisplit_weight > 0 and
    no noise_model_config is given."""
    import warnings

    from careamics.config.configuration_factories import (
        create_microsplit_configuration,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        create_microsplit_configuration(
            experiment_name="test",
            data_type="array",
            axes="SYX",
            patch_size=[64, 64],
            batch_size=4,
            denoisplit_weight=0.9,
            musplit_weight=0.1,
            output_channels=2,
        )

    captured = capsys.readouterr()
    assert "REMINDER" in captured.out or "noise" in captured.out.lower(), (
        "Expected a reminder about noise model but got none"
    )


# ---------------------------------------------------------------------------
# VAEModule.set_noise_model – extended tests
# ---------------------------------------------------------------------------


@pytest.fixture
def microsplit_module(tmp_path):
    """Minimal VAEModule with denoisplit_weight=0.9 (2 output channels)."""
    import numpy as np
    import warnings

    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
    from careamics.lightning.lightning_module import VAEModule

    weights = np.random.randn(3, 2).astype(np.float32)
    nm_cfg = GaussianMixtureNMConfig(
        weight=weights,
        min_signal=0.0,
        max_signal=255.0,
        min_sigma=125.0,
        n_gaussian=1,
        n_coeff=2,
    )
    mc_config = MultiChannelNMConfig(noise_models=[nm_cfg, nm_cfg])

    from careamics.config.configuration_factories import (
        create_microsplit_configuration,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = create_microsplit_configuration(
            experiment_name="test",
            data_type="array",
            axes="SYX",
            patch_size=[64, 64],
            batch_size=4,
            denoisplit_weight=0.9,
            musplit_weight=0.1,
            output_channels=2,
            noise_model_config=mc_config,
        )

    module = VAEModule(config.algorithm_config)
    return module, tmp_path


def test_set_noise_model_accepts_multichannel_noise_model(microsplit_module) -> None:
    """set_noise_model() accepts a MultiChannelNoiseModel directly."""
    module, _ = microsplit_module
    mc_nm = module.noise_model  # already attached

    module.set_noise_model(mc_nm)
    assert isinstance(module.noise_model, MultiChannelNoiseModel)


def test_set_noise_model_accepts_multichannel_nm_config(microsplit_module) -> None:
    """set_noise_model() accepts a MultiChannelNMConfig."""
    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    module, _ = microsplit_module
    weights = np.random.randn(3, 2).astype(np.float32)
    nm_cfg = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, min_sigma=125.0,
        n_gaussian=1, n_coeff=2,
    )
    mc_config = MultiChannelNMConfig(noise_models=[nm_cfg, nm_cfg])

    module.set_noise_model(mc_config)
    assert isinstance(module.noise_model, MultiChannelNoiseModel)


def test_set_noise_model_accepts_paths(microsplit_module, tmp_path) -> None:
    """set_noise_model() accepts a list of .npz paths."""
    module, _ = microsplit_module

    gen = np.random.default_rng(7)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)
    paths = trainer.save(tmp_path)

    module.set_noise_model([str(p) for p in paths])
    assert isinstance(module.noise_model, MultiChannelNoiseModel)
    assert module.noise_model._nm_cnt == 2


def test_set_noise_model_rejects_wrong_channel_count(microsplit_module) -> None:
    """set_noise_model() raises when channel count mismatches output_channels."""
    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig

    module, _ = microsplit_module
    weights = np.random.randn(3, 2).astype(np.float32)
    nm_cfg = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, min_sigma=125.0,
        n_gaussian=1, n_coeff=2,
    )
    mc_config = MultiChannelNMConfig(noise_models=[nm_cfg])  # only 1 instead of 2

    with pytest.raises(ValueError, match="Number of noise models"):
        module.set_noise_model(mc_config)


def test_set_noise_model_rejects_when_no_noise_model_required() -> None:
    """set_noise_model() raises when denoisplit_weight == 0 (musplit only)."""
    import warnings

    from careamics.config.noise_model import GaussianMixtureNMConfig
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
    from careamics.config.configuration_factories import (
        create_microsplit_configuration,
    )
    from careamics.lightning.lightning_module import VAEModule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = create_microsplit_configuration(
            experiment_name="musplit_test",
            data_type="array",
            axes="SYX",
            patch_size=[64, 64],
            batch_size=4,
            musplit_weight=1.0,
            denoisplit_weight=0.0,
            output_channels=2,
        )

    module = VAEModule(config.algorithm_config)

    weights = np.random.randn(3, 2).astype(np.float32)
    nm_cfg = GaussianMixtureNMConfig(
        weight=weights, min_signal=0.0, max_signal=255.0, min_sigma=125.0,
        n_gaussian=1, n_coeff=2,
    )
    mc_config = MultiChannelNMConfig(noise_models=[nm_cfg, nm_cfg])

    with pytest.raises(ValueError, match="denoisplit_weight <= 0"):
        module.set_noise_model(mc_config)


def test_set_noise_model_path_channel_order_mismatch_raises(
    microsplit_module, tmp_path
) -> None:
    """Issue #851: set_noise_model() raises when path channel_index is wrong."""
    module, _ = microsplit_module

    gen = np.random.default_rng(8)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=5)
    paths = trainer.save(tmp_path)

    # Supply the paths in reversed order — channel_index metadata says 0 but
    # we supply it at position 1, and vice versa
    reversed_paths = [str(paths[1]), str(paths[0])]
    with pytest.raises(ValueError, match="channel_index"):
        module.set_noise_model(reversed_paths)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnose_returns_per_channel_dicts() -> None:
    """diagnose() returns one dict per channel with expected keys."""
    gen = np.random.default_rng(9)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 20, signal.shape).astype(np.float32)

    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=10)

    diagnostics = trainer.diagnose(signal=signal, observation=observation)

    assert len(diagnostics) == 2
    expected_keys = {
        "channel_index",
        "final_loss",
        "loss_trend",
        "has_nan_weights",
        "has_inf_weights",
        "learned_sigma_mean",
        "signal_range_coverage",
        "wasserstein_distance",
    }
    for d in diagnostics:
        assert expected_keys == set(d.keys()), f"Unexpected keys: {set(d.keys())}"
        assert d["channel_index"] in (0, 1)
        assert not d["has_nan_weights"]
        assert not d["has_inf_weights"]
        assert 0.0 <= d["signal_range_coverage"] <= 1.0


def test_diagnose_without_training_raises() -> None:
    """diagnose() raises before training."""
    trainer = NoiseModelTrainer()
    signal = np.random.rand(4, 2, 16, 16).astype(np.float32)
    observation = signal + 0.1

    with pytest.raises(ValueError, match="No noise models available"):
        trainer.diagnose(signal=signal, observation=observation)


def test_train_losses_stored_after_training() -> None:
    """train_losses are populated with correct shape after training."""
    gen = np.random.default_rng(10)
    signal = gen.uniform(0, 255, (4, 2, 16, 16)).astype(np.float32)
    observation = signal + gen.normal(0, 10, signal.shape).astype(np.float32)

    n_epochs = 8
    trainer = NoiseModelTrainer(n_gaussian=1, n_coeff=2)
    trainer.train_from_pairs(signal=signal, observation=observation, n_epochs=n_epochs)

    assert trainer.train_losses is not None
    assert len(trainer.train_losses) == 2
    for losses in trainer.train_losses:
        assert len(losses) == n_epochs
        assert all(isinstance(v, float) for v in losses)
