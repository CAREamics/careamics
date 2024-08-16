"""Noise models config."""

from pathlib import Path
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class GaussianMixtureNmModel(BaseModel):
    """Gaussian mixture noise model."""

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="allow"
    )
    # TODO What are all these parameters?
    model_type: Literal["GaussianMixtureNoiseModel"]
    path: Union[str, Path] = None
    weight: Any = None  # TODO wtf ?
    n_gaussian: int = Field(default=1, ge=1)
    n_coeff: int = Field(default=2, ge=2)
    min_signal: float = Field(default=0.0, ge=0.0)
    max_signal: float = Field(default=1.0, ge=0.0)
    min_sigma: Any = Field(default=200.0, ge=0.0) # TODO took from nb in pn2v
    tol: float = Field(default=1e-10)  # TODO whatever the fuck this is
