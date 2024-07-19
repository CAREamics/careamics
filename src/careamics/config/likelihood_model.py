"""Likelihood model."""

from typing import Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn


class GaussianLikelihoodModel(BaseModel):
    """Gaussion likelihood model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),  # to allow using model_type
    )

    # TODO consider changing name to not collide with pydantic model_ namespace
    model_type: Literal["GaussianLikelihoodModel"]

    color_channels: int  # TODO output channels, rename
    ch_in: int = Field(default=64)  # input to the likelihood model
    predict_logvar: Literal[None, "pixelwise", "global", "channelwise"] = None
    logvar_lowerbound: float = None
    conv2d_bias: bool = True


class NMLikelihoodModel(BaseModel):
    """Likelihood model for noise model.

    Parameters
    ----------
    BaseModel
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    type: Literal["NMLikelihoodModel"]

    ch_in: int
    color_channels: int
    data_mean: Union[dict[str, torch.Tensor], torch.Tensor]
    data_std: Union[dict[str, torch.Tensor], torch.Tensor]
    noiseModel: nn.Module
