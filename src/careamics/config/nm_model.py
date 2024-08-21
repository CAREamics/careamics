"""Noise models config."""

from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


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
    n_coeff: int = Field(default=2, ge=2)
    min_signal: float = Field(default=0.0, ge=0.0)
    max_signal: float = Field(default=1.0, ge=0.0)
    min_sigma: Any = Field(default=200.0, ge=0.0)  # TODO took from nb in pn2v
    tol: float = Field(default=1e-10)  # TODO whatever the fuck this is

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
