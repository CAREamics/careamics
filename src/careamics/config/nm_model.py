"""Noise models config."""

from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

# TODO: add histogram-based noise model


class GaussianMixtureNmModel(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    # TODO What are all these parameters?
    model_type: Literal["GaussianMixtureNoiseModel"]
    path: Union[str, Path] = None
    signal: Union[str, Path, np.ndarray] = None
    observation: Union[str, Path, np.ndarray] = None
    weight: Any = None  # TODO wtf ?
    n_gaussian: int = Field(default=1, ge=1)
    """Number of gaussians in the mixture."""

    """Number of coefficients to describe the functional relationship between gaussian
    parameters and the signal. 2 implies a linear relationship, 3 implies a quadratic
    relationship and so on."""
    n_coeff: int = Field(default=2, ge=2)

    """Minimum signal intensity expected in the image."""
    min_signal: float = Field(default=0.0, ge=0.0)

    """Maximum signal intensity expected in the image."""
    max_signal: float = Field(default=1.0, ge=0.0)

    """All values of `standard deviation` below this are clamped to this value."""
    min_sigma: Any = Field(default=200.0, ge=0.0)  # TODO took from nb in pn2v

    """Tolerance used in the computation of the noise model likelihood."""
    tol: float = Field(default=1e-10)

    @model_validator(mode="after")  # TODO isn't it the best ever fucntion name ? :)
    def validate_path_to_pretrained_vs_data_to_train(self: Self):
        """_summary_.

        _extended_summary_

        Parameters
        ----------
        self : Self
            _description_
        """
        if (self.path and (self.signal is not None or self.observation is not None)):
            raise ValueError(
                "Either 'path' to pre-trained noise model should be"
                "provided or both signal and observation in form of paths"
                "or numpy arrays"
            )
        if not self.path and (self.signal is None or self.observation is None):
            raise ValueError(
                "Either 'path' to pre-trained noise model should be"
                "provided or both signal and observation in form of paths"
                "or numpy arrays"
            )


# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class MultiChannelNmModel(BaseModel):
    """Noise Model that aggregates the noise models for single channels."""

    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    """List of noise models, one for each target channel."""
    noise_models: list[GaussianMixtureNmModel]

