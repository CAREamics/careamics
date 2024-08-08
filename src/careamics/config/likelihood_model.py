"""Likelihood model."""

from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel

class GaussianLikelihoodModel(BaseModel):
    """Gaussion likelihood model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True)

    predict_logvar: Literal[None, "pixelwise"] = None
    logvar_lowerbound: float = None


class NMLikelihoodModel(BaseModel):
    """Likelihood model for noise model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_mean: Union[dict[str, torch.Tensor], torch.Tensor] = {"target": 0.0}
    data_std: Union[dict[str, torch.Tensor], torch.Tensor] = {"target": 0.0}
    noise_model: Union[GaussianMixtureNoiseModel, None] = None
