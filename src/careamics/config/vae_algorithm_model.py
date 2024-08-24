"""Algorithm configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from careamics.config.support import SupportedAlgorithm, SupportedLoss

from .architectures import CustomModel, LVAEModel
from .likelihood_model import GaussianLikelihoodConfig, NMLikelihoodConfig
from .nm_model import MultiChannelNMConfig
from .optimizer_models import LrSchedulerModel, OptimizerModel


class VAEAlgorithmConfig(BaseModel):
    """Algorithm configuration.

    # TODO

    Examples
    --------
    # TODO add once finalized
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
        extra="allow",
    )

    # Mandatory fields
    # defined in SupportedAlgorithm
    # TODO: Use supported Enum classes for typing?
    #   - values can still be passed as strings and they will be cast to Enum
    algorithm: Literal["musplit", "denoisplit"]
    loss: Literal["musplit", "denoisplit"]
    model: Union[LVAEModel, CustomModel] = Field(discriminator="architecture")

    noise_model: Optional[MultiChannelNMConfig] = None
    noise_model_likelihood_model: Optional[NMLikelihoodConfig] = None
    gaussian_likelihood_model: Optional[GaussianLikelihoodConfig] = None

    # Optional fields
    optimizer: OptimizerModel = OptimizerModel()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerModel = LrSchedulerModel()

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model based on `algorithm`.

        Returns
        -------
        Self
            The validated model.
        """
        # musplit
        if self.algorithm == SupportedAlgorithm.MUSPLIT:
            if self.loss != SupportedLoss.MUSPLIT:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `musplit`."
                )
            if self.model.predict_logvar != "pixelwise":
                raise ValueError(
                    "Algorithm `musplit` only supports `predict_logvar` as `pixelwise`."
                )
        # TODO add more checks

        if self.algorithm == SupportedAlgorithm.DENOISPLIT:
            if self.loss == SupportedLoss.DENOISPLIT:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `denoisplit`."
                )
            if self.model.predict_logvar is not None:
                raise ValueError(
                    "Algorithm `denoisplit` only supports `predict_logvar` as `None`."
                )

        # TODO: what if algorithm is not musplit or denoisplit
        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
