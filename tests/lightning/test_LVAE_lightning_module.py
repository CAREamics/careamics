from contextlib import nullcontext as does_not_raise
from typing import Callable

import pytest
import torch

from careamics.config.architectures import LVAEModel
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.likelihood_model import GaussianLikelihoodConfig, NMLikelihoodConfig
from careamics.models.lvae.likelihoods import GaussianLikelihood
from careamics.config import VAEAlgorithmConfig
from careamics.lightning import VAEModule
from careamics.losses import musplit_loss, denoisplit_loss, denoisplit_musplit_loss

@pytest.mark.parametrize(
    "multiscale_count, predict_logvar, loss_type, exp_error",
    [
        (1, None, 'musplit', does_not_raise()),
        (1, "pixelwise", 'musplit', does_not_raise()),
        (5, None, 'musplit', does_not_raise()),
        (5, "pixelwise", 'musplit', does_not_raise()),
        (1, None, 'denoisplit', pytest.raises(ValueError)),
    ],
)
def test_musplit_lightining_init(
    multiscale_count: int,
    predict_logvar: str,
    loss_type: str,
    exp_error: Callable,
):
    lvae_config = LVAEModel(
        architecture="LVAE",
        input_shape=64,
        multiscale_count=multiscale_count,
        z_dims=[128, 128, 128, 128],
        output_channels=3,
        predict_logvar=predict_logvar,
    )

    likelihood_config = GaussianLikelihoodConfig(
        predict_logvar=predict_logvar,
        logvar_lowerbound=0.0,
    )

    with exp_error:
        vae_config = VAEAlgorithmConfig(
            algorithm_type="vae",
            algorithm="musplit",
            loss=loss_type,
            model=lvae_config,
            gaussian_likelihood_model=likelihood_config,
        )
        lightning_model = VAEModule(
            algorithm_config=vae_config,
        )
        assert lightning_model is not None
        assert isinstance(lightning_model.model, torch.nn.Module)
        assert lightning_model.noise_model is None
        assert lightning_model.noise_model_likelihood is None
        assert isinstance(lightning_model.gaussian_likelihood, GaussianLikelihood)
        assert lightning_model.loss_func == musplit_loss


@pytest.mark.skip(reason="Not implemented yet")
def test_denoisplit_lightining_init():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_musplit_training_step():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_musplit_validation_step():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_musplit_logging():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_denoisplit_training_step():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_denoisplit_validation_step():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_denoisplit_logging():
    pass
