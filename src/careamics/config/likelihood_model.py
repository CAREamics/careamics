"""Likelihood model."""

from typing import Literal, Union, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn


class GaussianLikelihoodModel(BaseModel):
    """Gaussion likelihood model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True)

    color_channels: int = Field(default=2, ge=1)  # TODO output channels, rename, vals?
    ch_in: int = Field(default=64)  # input to the likelihood model
    predict_logvar: Literal[None, "pixelwise"] = None
    logvar_lowerbound: float = None
    conv2d_bias: bool = True


class NMLikelihoodModel(BaseModel):
    """Likelihood model for noise model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    color_channels: int = Field(default=2, ge=1)
    ch_in: int = Field(default=64)  # input to the likelihood model
    data_mean: Union[dict[str, torch.Tensor], torch.Tensor] = {"target": 0.0}
    data_std: Union[dict[str, torch.Tensor], torch.Tensor] = {"target": 0.0}
    noise_model: Optional[nn.Module] = None
