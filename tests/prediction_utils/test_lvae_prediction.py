import pytest
import torch

from careamics.models.lvae.likelihoods import GaussianLikelihood
from careamics.models.lvae.lvae import LadderVAE
from careamics.prediction_utils.lvae_prediction import (
    lvae_predict_mmse,
    lvae_predict_single_sample,
)


@pytest.fixture
def minimum_lvae_params():
    return {
        "input_shape": 64,
        "output_channels": 2,
        "multiscale_count": None,
        "z_dims": [128, 128, 128, 128],
        "encoder_n_filters": 64,
        "decoder_n_filters": 64,
        "encoder_dropout": 0.1,
        "decoder_dropout": 0.1,
        "nonlinearity": "ELU",
        "predict_logvar": "pixelwise",
        "enable_noise_model": False,
        "analytical_kl": False,
    }


@pytest.fixture
def gaussian_likelihood_params():
    return {"predict_logvar": "pixelwise", "logvar_lowerbound": -5}


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_single_sample(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test predictions of a single sample."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    # prediction
    y, log_var = lvae_predict_single_sample(model, likelihood_obj, x)

    assert y.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var is None


@pytest.mark.parametrize("predict_logvar", ["pixelwise", None])
@pytest.mark.parametrize("output_channels", [2, 3])
def test_lvae_predict_mmse(
    minimum_lvae_params, gaussian_likelihood_params, predict_logvar, output_channels
):
    """Test MMSE prediction."""
    minimum_lvae_params["predict_logvar"] = predict_logvar
    minimum_lvae_params["output_channels"] = output_channels
    gaussian_likelihood_params["predict_logvar"] = predict_logvar

    input_shape = minimum_lvae_params["input_shape"]

    # initialize model
    model = LadderVAE(**minimum_lvae_params)
    # initialize likelihood
    likelihood_obj = GaussianLikelihood(**gaussian_likelihood_params)

    # dummy input
    x = torch.rand(size=(1, 1, input_shape, input_shape))
    # prediction
    y, log_var = lvae_predict_mmse(model, likelihood_obj, x, mmse_count=5)

    assert y.shape == (1, output_channels, input_shape, input_shape)
    if predict_logvar == "pixelwise":
        assert log_var.shape == (1, output_channels, input_shape, input_shape)
    elif predict_logvar is None:
        assert log_var is None
