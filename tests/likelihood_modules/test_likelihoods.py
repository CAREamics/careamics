from pathlib import Path
from typing import Union

import numpy as np
import pytest
import torch

from careamics.config.likelihood_model import GaussianLikelihoodModel, NMLikelihoodModel
from careamics.config.nm_model import GaussianMixtureNmModel
from careamics.models.lvae.likelihoods import likelihood_factory
from careamics.models.lvae.noise_models import noise_model_factory

# TODO: move it under models/lvae/ ??

@pytest.mark.parametrize("target_ch", [1, 3])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
def test_gaussian_likelihood_output(
    target_ch: int, 
    predict_logvar: Union[str, None]
):
    config = GaussianLikelihoodModel(
        model_type="GaussianLikelihoodModel",
        predict_logvar=predict_logvar,
    )
    likelihood = likelihood_factory(config)
    
    img_size = 64
    inp_ch = target_ch * (1 + int(predict_logvar is not None))
    reconstruction = torch.rand((1, inp_ch, img_size, img_size)) 
    target = torch.rand((1, target_ch, img_size, img_size))
    out, _ = likelihood(reconstruction, target)
    
    exp_out_shape = (1, target_ch, img_size, img_size) 
    assert out.shape == exp_out_shape
    assert out[0].mean() is not None


@pytest.mark.skip(reason="Not implemented yet")
def test_nm_likelihood(tmp_path):
    config = NMLikelihoodModel(model_type="NMLikelihoodModel")
    inputs = torch.rand(1, 1, 64, 64)

    config.data_mean = {"target": inputs.mean()}
    config.data_std = {"target": inputs.std()}

    # define noise model
    nm_config = GaussianMixtureNmModel(model_type="GaussianMixtureNoiseModel")
    trained_weight = np.random.rand(18, 4)
    min_signal = np.random.rand(1)
    max_signal = np.random.rand(1)
    min_sigma = np.random.rand(1)
    filename = Path(tmp_path) / "gm_noise_model.npz"
    np.savez(
        filename,
        trained_weight=trained_weight,
        min_signal=min_signal,
        max_signal=max_signal,
        min_sigma=min_sigma,
    )
    config.noise_model = noise_model_factory(nm_config, [filename])
    likelihood = likelihood_factory(config)
    assert likelihood(inputs, inputs)[0].mean() is not None
    # TODO add more meaningful tests
