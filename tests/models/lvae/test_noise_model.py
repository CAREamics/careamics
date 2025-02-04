from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.likelihoods import NoiseModelLikelihood
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    noise_model_factory,
)

pytestmark = pytest.mark.lvae


def test_factory_no_noise_model():
    noise_model = noise_model_factory(None)
    assert noise_model is None


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
    noise_model = noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125


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
    noise_model = noise_model_factory(noise_model_config)
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
    nm = noise_model_factory(noise_model_config)
    assert nm is not None
    assert isinstance(nm, MultiChannelNoiseModel)
    assert nm._nm_cnt == target_ch
    assert all(
        isinstance(getattr(nm, f"nmodel_{i}"), GaussianMixtureNoiseModel)
        for i in range(nm._nm_cnt)
    )
    assert all(
        np.allclose(
            getattr(nm, f"nmodel_{i}").weight,
            create_dummy_noise_model["trained_weight"] + rand_epss[i],
        )
        for i in range(nm._nm_cnt)
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
    test_input = torch.rand(256, 256)
    test_target = torch.rand(256, 256)

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel", n_gaussian=1
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    likelihood = NoiseModelLikelihood(
        data_mean=test_input.mean(), data_std=test_input.std(), noise_model=noise_model
    )
    log_likelihood, _ = likelihood(test_input, test_target)
    assert log_likelihood is not None
