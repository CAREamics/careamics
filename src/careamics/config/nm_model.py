"""Noise models config."""

from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# TODO: add histogram-based noise model


class GaussianMixtureNmModel(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    model_type: Literal["GaussianMixtureNoiseModel"]  # TODO: why do we need this?
    """Model type."""

    path: Union[Path, str, None] = None
    """Path to the directory where the trained noise model (*.npz) is saved in the
    `train` method."""

    weight: Any = None
    """A [3*n_gaussian, n_coeff] sized array containing the values of the weights
    describing the GMM noise model, with each row corresponding to one
    parameter of each gaussian, namely [mean, standard deviation and weight].
    Specifically, rows are organized as follows:
    - first n_gaussian rows correspond to the means
    - next n_gaussian rows correspond to the weights
    - last n_gaussian rows correspond to the standard deviations
    If `weight=None`, the weight array is initialized using the `min_signal`
    and `max_signal` parameters."""

    n_gaussian: int = Field(default=1, ge=1)
    """Number of gaussians in the mixture."""

    n_coeff: int = Field(default=2, ge=2)
    """Number of coefficients to describe the functional relationship between gaussian
    parameters and the signal. 2 implies a linear relationship, 3 implies a quadratic
    relationship and so on."""

    min_signal: float = Field(default=0.0, ge=0.0)
    """Minimum signal intensity expected in the image."""

    max_signal: float = Field(default=1.0, ge=0.0)
    """Maximum signal intensity expected in the image."""

    min_sigma: Any = None
    """All values of `standard deviation` below this are clamped to this value."""

    tol: float = Field(default=1e-10)
    """Tolerance used in the computation of the noise model likelihood."""


# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class MultiChannelNmModel(BaseModel):
    """Noise Model that aggregates the noise models for single channels."""

    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    noise_models: list[GaussianMixtureNmModel]
    """List of noise models, one for each target channel."""
