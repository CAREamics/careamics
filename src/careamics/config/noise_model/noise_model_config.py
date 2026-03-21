"""Noise models config."""

from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
)

from careamics.utils.serializers import _array_to_json, _to_numpy

# TODO: this is a temporary solution to serialize and deserialize array fields
# in pydantic models. Specifically, the aim is to enable saving and loading configs
# with such arrays to/from JSON files during, resp., training and evaluation.
Array = Annotated[
    Union[np.ndarray, torch.Tensor],
    PlainSerializer(_array_to_json, return_type=str),
    PlainValidator(_to_numpy),
]
"""Annotated array type, used to serialize arrays or tensors to JSON strings
and deserialize them back to arrays."""


# TODO: add histogram-based noise model


class GaussianMixtureNMConfig(BaseModel):
    """Gaussian mixture noise model configuration.

    Weights can be loaded from a previously saved `.npz` file using the
    `from_npz` classmethod, which populates the `weight`, `min_signal`,
    `max_signal`, and `min_sigma` fields directly.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="allow",
    )

    model_type: Literal["GaussianMixtureNoiseModel"] = "GaussianMixtureNoiseModel"

    weight: Array | None = None
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
    """Number of gaussians used for the GMM."""

    n_coeff: int = Field(default=2, ge=2)
    """Number of coefficients to describe the functional relationship between gaussian
    parameters and the signal. 2 implies a linear relationship, 3 implies a quadratic
    relationship and so on."""

    min_signal: float = Field(default=0.0, ge=0.0)
    """Minimum signal intensity expected in the image."""

    max_signal: float = Field(default=1.0, ge=0.0)
    """Maximum signal intensity expected in the image."""

    min_sigma: float = Field(default=125.0, ge=0.0)
    """Minimum value of `standard deviation` allowed in the GMM.
    All values of `standard deviation` below this are clamped to this value."""

    tol: float = Field(default=1e-10)
    """Tolerance used in the computation of the noise model likelihood."""

    @classmethod
    def from_npz(cls, path: Union[str, Path]) -> "GaussianMixtureNMConfig":
        """Load a trained Gaussian mixture noise model from a `.npz` file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the `.npz` file produced by `GaussianMixtureNoiseModel.save`.

        Returns
        -------
        GaussianMixtureNMConfig
            Configuration populated with weights and signal range from the file.

        Raises
        ------
        ValueError
            If the path does not exist or does not point to a `.npz` file.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")
        if path.suffix != ".npz":
            raise ValueError(f"Path {path} must point to a .npz file.")
        if not path.is_file():
            raise ValueError(f"Path {path} must point to a file.")
        params = np.load(path)
        weight_key = "trained_weight" if "trained_weight" in params else "weight"
        return cls(
            weight=params[weight_key],
            min_signal=float(params["min_signal"]),
            max_signal=float(params["max_signal"]),
            min_sigma=float(params["min_sigma"]),
        )


# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class MultiChannelNMConfig(BaseModel):
    """Noise Model config aggregating noise models for single output channels."""

    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    noise_models: list[GaussianMixtureNMConfig]
    """List of noise models, one for each target channel."""
