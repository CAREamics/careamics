"""Likelihood model."""

from typing import Literal, Union

import torch
from pydantic import BaseModel, ConfigDict

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel, MultiChannelNoiseModel
)
    
NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]

class GaussianLikelihoodModel(BaseModel):
    """Gaussion likelihood model.

    Parameters
    ----------
    predict_logvar: Union[Literal["pixelwise"], None], optional
        If `pixelwise`, log-variance is computed for each pixel, else log-variance
        is not computed. Default is `None`.
    logvar_lowerbound: float, optional
        The lowerbound value for log-variance. Default is `None`.
    """

    model_config = ConfigDict(validate_assignment=True)

    predict_logvar: Literal[None, "pixelwise"] = None # TODO: is the type correct?
    logvar_lowerbound: Union[float, None] = None


class NMLikelihoodModel(BaseModel):
    """Likelihood model for noise model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_mean: Union[torch.Tensor] = torch.zeros(1)
    data_std: Union[torch.Tensor] = torch.ones(1)
    noise_model: Union[NoiseModel, None] = None
