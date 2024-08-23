from pathlib import Path

import numpy as np
import pytest
import torch

from careamics.config import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    noise_model_factory,
)


# TODO: move to conftest.py as pytest.fixture
def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    weights = np.random.rand(3 * n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)


@pytest.mark.skip(reason="Conflicting with the current implementation")
def test_factory_no_noise_model():
    noise_model_config = MultiChannelNMConfig(noise_models=[])
    noise_model = noise_model_factory(noise_model_config)
    assert noise_model is None


def test_instantiate_noise_model(tmp_path: Path) -> None:
    # Create a dummy noise model
    create_dummy_noise_model(tmp_path, 3, 3)

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


def test_instantiate_multiple_noise_models(tmp_path: Path) -> None:
    # Create a dummy noise model
    create_dummy_noise_model(tmp_path, 3, 3)

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


@pytest.mark.parametrize("n_gaussians", [3, 5])
@pytest.mark.parametrize("n_coeffs", [2, 4])
@pytest.mark.parametrize("img_size", [64, 128])
def test_noise_model_likelihood(
    tmp_path: Path, n_gaussians: int, n_coeffs: int, img_size: int
) -> None:
    create_dummy_noise_model(tmp_path, n_gaussians, n_coeffs)

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


@pytest.mark.parametrize("n_gaussians", [3, 5])
@pytest.mark.parametrize("n_coeffs", [2, 4])
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("target_ch", [1, 3, 5])
def test_multi_channel_noise_model_likelihood(
    tmp_path: Path, n_gaussians: int, n_coeffs: int, img_size: int, target_ch: int
) -> None:
    create_dummy_noise_model(tmp_path, n_gaussians, n_coeffs)

    gmm = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path / "dummy_noise_model.npz",
        # all other params are default
    )
    noise_model_config = MultiChannelNMConfig(noise_models=[gmm] * target_ch)
    nm = noise_model_factory(noise_model_config)
    assert nm is not None
    assert isinstance(nm, MultiChannelNoiseModel)
    assert nm._nm_cnt == target_ch
    assert all(
        isinstance(getattr(nm, f"nmodel_{i}"), GaussianMixtureNoiseModel)
        for i in range(nm._nm_cnt)
    )

    inp_shape = (1, target_ch, img_size, img_size)
    signal = torch.ones(inp_shape)
    obs = signal + torch.randn(inp_shape) * 0.1
    likelihood = nm.likelihood(obs, signal)
    assert likelihood.shape == inp_shape


@pytest.mark.skip(reason="Need to refac noise model to be able to train on CPU")
def test_gm_noise_model_training(tmp_path):
    x = np.random.rand(3)
    y = np.random.rand(3)

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel", signal=x, observation=y
    )

    noise_model = GaussianMixtureNoiseModel(nm_config)

    # Test training
    output = noise_model.train(x, y, n_epochs=2)
    assert output is not None
    # TODO do something with output ?
