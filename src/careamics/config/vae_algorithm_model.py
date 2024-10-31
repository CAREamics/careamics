"""Algorithm configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from careamics.config.support import SupportedAlgorithm, SupportedLoss

from .architectures import CustomModel, LVAEModel
from .likelihood_model import GaussianLikelihoodConfig, NMLikelihoodConfig
from .loss_model import LVAELossConfig
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

    # NOTE: these are all configs (pydantic models)s
    model: Union[LVAEModel, CustomModel] = Field(discriminator="architecture")
    loss: Optional[LVAELossConfig]
    noise_model: Optional[MultiChannelNMConfig] = None
    noise_model_likelihood: Optional[NMLikelihoodConfig] = None
    gaussian_likelihood: Optional[GaussianLikelihoodConfig] = None

    # Optional fields
    optimizer: Optional[OptimizerModel] = OptimizerModel()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: Optional[LrSchedulerModel] = LrSchedulerModel()

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
            if self.loss.loss_type != SupportedLoss.MUSPLIT:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `musplit`."
                )

        if self.algorithm == SupportedAlgorithm.DENOISPLIT:
            if self.loss.loss_type not in [
                SupportedLoss.DENOISPLIT,
                SupportedLoss.DENOISPLIT_MUSPLIT,
            ]:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `denoisplit` "
                    "or `denoisplit_musplit."
                )
            if (
                self.loss.loss_type == SupportedLoss.DENOISPLIT
                and self.model.predict_logvar is not None
            ):
                raise ValueError(
                    "Algorithm `denoisplit` with loss `denoisplit` only supports "
                    "`predict_logvar` as `None`."
                )
            if self.noise_model is None:
                raise ValueError("Algorithm `denoisplit` requires a noise model.")
        # TODO: what if algorithm is not musplit or denoisplit (HDN?)
        return self

    @model_validator(mode="after")
    def output_channels_validation(self: Self) -> Self:
        """Validate the consistency between number of out channels and noise models.

        Returns
        -------
        Self
            The validated model.
        """
        if self.noise_model is not None:
            assert self.model.output_channels == len(self.noise_model.noise_models), (
                f"Number of output channels ({self.model.output_channels}) must match "
                f"the number of noise models ({len(self.noise_model.noise_models)})."
            )
        return self

    @model_validator(mode="after")
    def predict_logvar_validation(self: Self) -> Self:
        """Validate the consistency of `predict_logvar` throughout the model.

        Returns
        -------
        Self
            The validated model.
        """
        if self.gaussian_likelihood is not None:
            assert (
                self.model.predict_logvar == self.gaussian_likelihood.predict_logvar
            ), (
                f"Model `predict_logvar` ({self.model.predict_logvar}) must match "
                "Gaussian likelihood model `predict_logvar` "
                f"({self.gaussian_likelihood.predict_logvar}).",
            )
        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
