from pathlib import Path

import pytest
import numpy as np
import torch

from careamics.models.lvae.noise_models import noise_model_factory
from careamics.config import GaussianMixtureNmModel, NMModel

def create_dummy_noise_model(
    tmp_path: Path,
    n_gaussians: int = 3,
    n_coeffs: int = 3,
) -> None:
    weights = np.random.rand(3*n_gaussians, n_coeffs)
    nm_dict = {
        "trained_weight": weights,
        "min_signal": np.array([0]),
        "max_signal": np.array([2**16 - 1]),
        "min_sigma": 0.125,
    }
    np.savez(tmp_path / "dummy_noise_model.npz", **nm_dict)
    

def test_instantiate_noise_model(
    tmp_path: Path
) -> None:
    # Create a dummy noise model
    create_dummy_noise_model(tmp_path, 3, 3)
    
    # Instantiate the noise model
    gmm = GaussianMixtureNmModel(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path  / "dummy_noise_model.npz",
        # all other params are default
    )  
    noise_model_config = NMModel(
        noise_models=[gmm]
    )
    noise_model = noise_model_factory(noise_model_config)
    assert noise_model is not None
    assert noise_model.nmodel_0.weight.shape == (9, 3)
    assert noise_model.nmodel_0.min_signal == 0
    assert noise_model.nmodel_0.max_signal == 2**16 - 1
    assert noise_model.nmodel_0.min_sigma == 0.125
    
def test_instantiate_multiple_noise_models(
    tmp_path: Path
) -> None:
    # Create a dummy noise model
    create_dummy_noise_model(tmp_path, 3, 3)
    
    # Instantiate the noise model
    gmm = GaussianMixtureNmModel(
        model_type="GaussianMixtureNoiseModel",
        path=tmp_path  / "dummy_noise_model.npz",
        # all other params are default
    )  
    noise_model_config = NMModel(
        noise_models=[gmm, gmm, gmm]
    )
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