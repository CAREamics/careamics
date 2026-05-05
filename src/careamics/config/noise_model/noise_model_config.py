"""Noise models config."""

from pathlib import Path
from typing import Annotated, Literal, Self, Union

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    model_validator,
)

from careamics.utils.serializers import _array_to_json, _to_numpy, _to_torch

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

Tensor = Annotated[
    torch.Tensor,
    PlainSerializer(_array_to_json, return_type=str),
    PlainValidator(_to_torch),
]
"""Annotated tensor type, used to serialize tensors to JSON strings
and deserialize them back to tensors."""


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

    min_sigma: float = Field(default=200.0, ge=0.0)
    """Minimum value of `standard deviation` allowed in the GMM.
    All values of `standard deviation` below this are clamped to this value."""

    tol: float = Field(default=1e-10)
    """Tolerance used in the computation of the noise model likelihood."""

    channel_index: int | None = Field(default=None, ge=0)
    """The data channel index this noise model was trained on.
    Used to validate channel ordering when attaching to a multi-channel model."""

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
            `channel_index` is populated when present in the file.

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
        channel_index = (
            int(params["channel_index"]) if "channel_index" in params else None
        )
        return cls(
            weight=params[weight_key],
            min_signal=float(params["min_signal"]),
            max_signal=float(params["max_signal"]),
            min_sigma=float(params["min_sigma"]),
            channel_index=channel_index,
        )


# The noise model is given by a set of GMMs, one for each target
# e.g., 2 target channels, 2 noise models
class MultiChannelNMConfig(BaseModel):
    """Noise Model config aggregating noise models for single output channels.

    When `channel_indices` is provided it must agree with the `channel_index`
    stored on each contained `GaussianMixtureNMConfig` (when that field is set),
    and its length must equal the number of noise models.  This enables
    detecting channel-order swaps before training.
    """

    # TODO: check that this model config is OK
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    noise_models: list[GaussianMixtureNMConfig]
    """List of noise models, one for each target channel."""

    channel_indices: list[int] | None = None
    """Ordered list of data channel indices, noise_models[i] was trained on
    channel channel_indices[i].  When None the natural list order is assumed."""

    @model_validator(mode="after")
    def _validate_channel_order(self) -> Self:
        """Validate length/content and consistency of ``channel_indices``.

        Returns
        -------
        Self
            Validated model instance.
        """
        if self.channel_indices is not None:
            if len(self.channel_indices) != len(self.noise_models):
                raise ValueError(
                    f"channel_indices length ({len(self.channel_indices)}) must "
                    f"match the number of noise models ({len(self.noise_models)})."
                )
            for pos, (ci, nm) in enumerate(
                zip(self.channel_indices, self.noise_models, strict=True)
            ):
                if nm.channel_index is not None and nm.channel_index != ci:
                    raise ValueError(
                        f"Noise model at position {pos} has channel_index="
                        f"{nm.channel_index} but channel_indices[{pos}]={ci}. "
                        "Channel order mismatch detected."
                    )
            expected = list(range(len(self.noise_models)))
            if sorted(self.channel_indices) != expected:
                raise ValueError(
                    f"channel_indices must be a permutation of {expected}, "
                    f"got {self.channel_indices}."
                )
        return self
