"""VAE-based algorithm Pydantic model."""

from __future__ import annotations

from pprint import pformat
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import LVAELossConfig
from careamics.config.nm_model import MultiChannelNMConfig
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel
from careamics.config.support import SupportedAlgorithm, SupportedLoss


class VAEBasedAlgorithm(BaseModel):
    """VAE-based algorithm configuration.

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
    algorithm: Literal["hdn", "microsplit"]

    # NOTE: these are all configs (pydantic models)
    loss: LVAELossConfig
    model: LVAEModel
    noise_model: MultiChannelNMConfig | None = None
    noise_model_likelihood: NMLikelihoodConfig | None = None
    gaussian_likelihood: GaussianLikelihoodConfig | None = None  # TODO change to str

    mmse_count: int = 1
    is_supervised: bool = False

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
        # hdn
        # TODO move to designated configurations
        if self.algorithm == SupportedAlgorithm.HDN:
            if self.loss.loss_type != SupportedLoss.HDN:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `hdn`."
                )
            if self.model.multiscale_count > 1:
                raise ValueError("Algorithm `hdn` does not support multiscale models.")
        # musplit
        if self.algorithm == SupportedAlgorithm.MICROSPLIT:
            if self.loss.loss_type not in [
                SupportedLoss.MUSPLIT,
                SupportedLoss.DENOISPLIT,
                SupportedLoss.DENOISPLIT_MUSPLIT,
            ]:  # TODO Update losses configs, make loss just microsplit
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `microsplit`."
                )  # TODO Update losses configs

            if (
                self.loss.loss_type == SupportedLoss.DENOISPLIT
                and self.model.predict_logvar is not None
            ):
                raise ValueError(
                    "Algorithm `denoisplit` with loss `denoisplit` only supports "
                    "`predict_logvar` as `None`."
                )
            if (
                self.loss.loss_type == SupportedLoss.DENOISPLIT
                and self.noise_model is None
            ):
                raise ValueError("Algorithm `denoisplit` requires a noise model.")
        # TODO: what if algorithm is not musplit or denoisplit
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

        if self.algorithm == SupportedAlgorithm.HDN:
            assert self.model.output_channels == 1, (
                f"Number of output channels ({self.model.output_channels}) must be 1 "
                "for algorithm `hdn`."
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
        # if self.algorithm == SupportedAlgorithm.HDN:
        #     assert (
        #         self.model.predict_logvar is None
        #     ), "Model `predict_logvar` must be `None` for algorithm `hdn`."
        #     if self.gaussian_likelihood is not None:
        #         assert self.gaussian_likelihood.predict_logvar is None, (
        #             "Gaussian likelihood model `predict_logvar` must be `None` "
        #             "for algorithm `hdn`."
        #         )
        # TODO check this
        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    @classmethod
    def get_compatible_algorithms(cls) -> list[str]:
        """Get the list of compatible algorithms.

        Returns
        -------
        list of str
            List of compatible algorithms.
        """
        return ["hdn", "microsplit"]
