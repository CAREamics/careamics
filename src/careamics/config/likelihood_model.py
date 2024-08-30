"""Likelihood model."""

from typing import Literal, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator
from typing_extensions import Annotated

from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)
from careamics.utils.serializers import array_to_json, list_to_torch

NoiseModel = Union[GaussianMixtureNoiseModel, MultiChannelNoiseModel]

Tensor = Annotated[
    Union[np.ndarray, torch.Tensor], 
    PlainSerializer(array_to_json, return_type=str),
    PlainValidator(list_to_torch)
]
"""Annotated tensor type, used to serialize arrays or tensors to JSON strings
and deserialize them back to tensors."""


class GaussianLikelihoodConfig(BaseModel):
    """Gaussian likelihood configuration."""

    model_config = ConfigDict(validate_assignment=True)

    predict_logvar: Optional[Literal["pixelwise"]] = None
    """If `pixelwise`, log-variance is computed for each pixel, else log-variance
    is not computed."""

    logvar_lowerbound: Union[float, None] = None
    """The lowerbound value for log-variance."""


class NMLikelihoodConfig(BaseModel):
    """Noise model likelihood configuration."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # TODO remove and use as parameters to the likelihood functions?
    data_mean: Tensor = torch.zeros(1)
    """The mean of the data, used to unnormalize data for noise model evaluation.
    Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""

    # TODO remove and use as parameters to the likelihood functions?
    data_std: Tensor = torch.ones(1)
    """The standard deviation of the data, used to unnormalize data for noise
    model evaluation. Shape is (target_ch,) (or (1, target_ch, [1], 1, 1))."""

    # TODO: serialization/deserialization for this
    noise_model: Optional[NoiseModel] = Field(default=None, exclude=True)
    """The noise model instance used to compute the likelihood."""
