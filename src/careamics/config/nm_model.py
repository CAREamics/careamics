"""Noise models config."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GaussianMixtureNoiseModel(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # to allow using model_type as a name
    )
    # TODO What are all these parameters?
    model_type: Literal["GaussianMixtureNoiseModel"]  # TODO consider changing name
    weight: Any = None  # TODO wtf ?
    n_gaussian: int = Field(default=1, ge=1)
    n_coeff: int = Field(default=2, ge=2)
    min_signal: float = Field(default=0.0, ge=0.0)
    max_signal: float = Field(default=1.0, ge=0.0)
    min_sigma: Any = None
    tol: float = Field(default=1e-10)  # TODO whatever the fuck this is
