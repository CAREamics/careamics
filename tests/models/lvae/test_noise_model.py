from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    multichannel_noise_model_factory,
    noise_model_factory,
)

pytestmark = pytest.mark.lvae


def test_factory_no_noise_model():
    noise_model = multichannel_noise_model_factory(None)
    assert noise_model is None


@pytest.mark.skip(reason="Needs to be updated")
def test_instantiate_noise_model(tmp_path: Path, create_dummy_noise_model) -> None:
    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm])
    noise_model = multichannel_noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125


@pytest.mark.skip(reason="Needs to be updated")
def test_instantiate_multiple_noise_models(
    tmp_path: Path, create_dummy_noise_model
) -> None:
    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm, gmm, gmm])
    noise_model = multichannel_noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0 is not None
    assert noise_model.nmodel_1 is not None
    assert noise_model.nmodel_2 is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125
    assert noise_model.nmodel_1.weight.shape == (9, 3)
    assert noise_model.nmodel_1.min_signal == 0
    assert noise_model.nmodel_1.max_signal == 2**16 - 1
    assert noise_model.nmodel_1.min_sigma == 0.125
    assert noise_model.nmodel_2.weight.shape == (9, 3)
    assert noise_model.nmodel_2.min_signal == 0
    assert noise_model.nmodel_2.max_signal == 2**16 - 1
    assert noise_model.nmodel_2.min_sigma == 0.125


@pytest.mark.parametrize("img_size", [64, 128])
def test_noise_model_likelihood(
    tmp_path: Path,
    img_size: int,
    create_dummy_noise_model,
) -> None:
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    gmm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    nm = GaussianMixtureNoiseModel(gmm_config)
    assert nm is not None
    assert isinstance(nm, GaussianMixtureNoiseModel)

    inp_shape = (1, 1, img_size, img_size)
    signal = torch.ones(inp_shape)
    obs = signal + torch.randn(inp_shape) * 0.1
    likelihood = nm.likelihood(obs, signal)
    assert likelihood.shape == inp_shape


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("target_ch", [1, 2, 3])
def test_multi_channel_noise_model_likelihood(
    tmp_path: Path,
    img_size: int,
    target_ch: int,
    create_dummy_noise_model,
) -> None:
    noise_models = []
    rand_epss = []
    for i in range(target_ch):
        eps = np.random.rand()
        nm_dict = create_dummy_noise_model.copy()
        nm_dict["trained_weight"] = nm_dict["trained_weight"] + eps
        rand_epss.append(eps)
        np.savez(tmp_path / f"dummy_noise_model_{i}.npz", **nm_dict)

        gmm = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            path=tmp_path / f"dummy_noise_model_{i}.npz",
            # all other params are default
        )
        noise_models.append(gmm)

    noise_model_config = MultiChannelNMConfig(noise_models=noise_models)
    nm = multichannel_noise_model_factory(noise_model_config)
    assert nm is not None
    assert isinstance(nm, MultiChannelNoiseModel)
    assert nm._nm_count == target_ch
    assert all(
        isinstance(getattr(nm, f"nmodel_{i}"), GaussianMixtureNoiseModel)
        for i in range(nm._nm_count)
    )
    assert all(
        np.allclose(
            getattr(nm, f"nmodel_{i}").weight,
            create_dummy_noise_model["trained_weight"] + rand_epss[i],
        )
        for i in range(nm._nm_count)
    )
    inp_shape = (1, target_ch, img_size, img_size)
    signal = torch.ones(inp_shape)
    obs = signal + torch.randn(inp_shape) * 0.1
    likelihood = nm.likelihood(obs, signal)
    assert likelihood.shape == inp_shape


@pytest.mark.parametrize(
    "image_size, max_value, noise_scale",
    [
        ([5, 128, 128], 255, 0.1),
        ([5, 128, 128], 255, 0.5),
    ],
)
def test_gm_noise_model_training(image_size, max_value, noise_scale):
    gen = np.random.default_rng(42)
    signal_normalized = gen.uniform(0, 1, image_size)
    noise = gen.normal(0, noise_scale, image_size)
    observation_normalized = signal_normalized + noise
    signal = signal_normalized * max_value
    observation = observation_normalized * max_value

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        n_gaussian=1,
        min_signal=signal.min(),
        max_signal=signal.max(),
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    training_losses = noise_model.fit(
        signal=signal, observation=observation, n_epochs=500
    )
    initial_loss = training_losses[0]
    last_loss = training_losses[-1]
    # Check if model is training
    assert initial_loss > last_loss

    # check if estimated mean and std of a noisy sample are close to real ones
    signal_tensor = torch.from_numpy(signal).to(torch.float32)
    mus, sigmas, _ = noise_model.get_gaussian_parameters(signal_tensor)

    # learned mean should be close to the mean of the signal
    learned_mu = mus.mean() / max_value
    real_mu = signal_normalized.mean()
    assert np.allclose(learned_mu, real_mu, atol=1e-2)

    # learned sigma should be close to the noise sigma
    learned_sigma = sigmas.mean() / max_value
    noise_image = observation_normalized - signal_normalized
    real_sigma = noise_image.std()
    assert np.allclose(learned_sigma, real_sigma, atol=1e-2)


@pytest.mark.parametrize("image_size, max_value", [([256, 256], 255)])
def test_noise_model_sampling(image_size, max_value):
    gen = np.random.default_rng(42)

    signal = gen.uniform(0, 1, image_size)
    observation = signal + gen.normal(0, 0.1, signal.shape)
    signal = signal * max_value
    observation = observation * max_value

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        n_gaussian=1,
        min_sigma=100,
        min_signal=signal.min(),
        max_signal=signal.max(),
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    noise_model.fit(signal=signal, observation=observation, n_epochs=200)
    sampled_noise_data = noise_model.sample_observation_from_signal(signal)
    assert sampled_noise_data.shape == signal.shape

    real_noise = observation - signal
    predicted_noise = sampled_noise_data - signal
    real_noise = real_noise / max_value
    predicted_noise = predicted_noise / max_value
    noise_distribution_difference = wasserstein_distance(
        real_noise.ravel(), predicted_noise.ravel()
    )
    assert noise_distribution_difference < 0.1


def test_noise_model_in_likelihood_call():
    from careamics.losses.lvae.lvae_losses import _compute_noise_model_log_likelihood
    from careamics.models.lvae.noise_models import MultiChannelNoiseModel

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel", n_gaussian=1
    )
    single_nm = GaussianMixtureNoiseModel(nm_config)
    noise_model = MultiChannelNoiseModel([single_nm])

    test_input = torch.rand(1, 1, 64, 64)
    test_target = torch.rand(1, 1, 64, 64)
    data_mean = test_input.mean().item()
    data_std = test_input.std().item()
    normalized_noise_model = noise_model.get_normalized_copy([data_mean], [data_std])

    log_likelihood = _compute_noise_model_log_likelihood(
        reconstruction=test_input,
        target=test_target,
        noise_model=normalized_noise_model,
    )
    assert log_likelihood is not None
    assert isinstance(log_likelihood, torch.Tensor)


def test_single_noise_model_factory_no_config():
    """Test single-channel noise model factory with None input."""
    noise_model = noise_model_factory(None)
    assert noise_model is None


@pytest.mark.skip(reason="Needs to be updated")
def test_single_noise_model_factory_instantiate(
    tmp_path: Path, create_dummy_noise_model
) -> None:
    """Test instantiating a single noise model with single-channel factory."""
    # Create a dummy noise model
    np.savez(tmp_path / "dummy_noise_model.npz", **create_dummy_noise_model)

    # Instantiate the noise model using single GaussianMixtureNMConfig
    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model = noise_model_factory(gmm)
    assert noise_model is not None
    assert isinstance(noise_model, GaussianMixtureNoiseModel)
    assert noise_model.weight.shape == (9, 3)
    assert noise_model.min_signal == 0
    assert noise_model.max_signal == 2**16 - 1
    assert noise_model.min_sigma == 0.125


def _sorted_rows(pairs: torch.Tensor) -> np.ndarray:
    array = pairs.numpy()
    return array[np.lexsort((array[:, 1], array[:, 0]))]


def test_signal_observation_pairs_multiple_observations_per_signal():
    """Within a pair, consecutive observations share the same signal image."""
    gen = np.random.default_rng(42)
    signals = [gen.uniform(0, 255, (2, 8, 8)), gen.uniform(0, 255, (1, 4, 4))]
    # 3 observations per signal image in the first pair, 2 in the second
    observations = [
        np.repeat(signals[0], 3, axis=0) + gen.normal(0, 10, (6, 8, 8)),
        np.repeat(signals[1], 2, axis=0) + gen.normal(0, 10, (2, 4, 4)),
    ]

    config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        n_gaussian=1,
        min_signal=0,
        max_signal=255,
    )
    pairs = GaussianMixtureNoiseModel(config).get_signal_observation_pairs(
        signals, observations, lower_clip=0.0, upper_clip=100.0
    )

    expected = np.concatenate(
        [
            np.stack(
                [np.repeat(sig, len(obs) // len(sig), axis=0).ravel(), obs.ravel()],
                axis=1,
            )
            for sig, obs in zip(signals, observations, strict=True)
        ]
    ).astype(np.float32)
    signal_values = expected[:, 0]
    expected = expected[
        (signal_values > signal_values.min()) & (signal_values < signal_values.max())
    ]
    np.testing.assert_array_equal(
        _sorted_rows(pairs), _sorted_rows(torch.from_numpy(expected))
    )


def test_gm_noise_model_training_on_arrays_with_different_shapes():
    """Fitting on a sequence of differently shaped arrays learns the noise."""
    gen = np.random.default_rng(42)
    max_value = 255
    noise_scale = 0.1
    signals_normalized = [
        gen.uniform(0, 1, (3, 128, 128)),
        gen.uniform(0, 1, (2, 96, 160)),
        gen.uniform(0, 1, (1, 64, 64)),
    ]
    signals = [sig * max_value for sig in signals_normalized]
    observations = [
        (sig + gen.normal(0, noise_scale, sig.shape)) * max_value
        for sig in signals_normalized
    ]

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        n_gaussian=1,
        min_signal=min(sig.min() for sig in signals),
        max_signal=max(sig.max() for sig in signals),
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    training_losses = noise_model.fit(
        signal=signals, observation=observations, n_epochs=500
    )
    assert training_losses[0] > training_losses[-1]

    signal_tensor = torch.from_numpy(np.concatenate([s.ravel() for s in signals]))
    _, sigmas, _ = noise_model.get_gaussian_parameters(signal_tensor.float())
    learned_sigma = sigmas.mean() / max_value
    assert np.allclose(learned_sigma, noise_scale, atol=1e-2)
