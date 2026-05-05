"""VAE-based algorithm Pydantic model."""

from __future__ import annotations

import warnings
from pprint import pformat
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from careamics.config.architectures import LVAEConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)
from careamics.config.losses.loss_config import LVAELossConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
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
    model: LVAEConfig
    noise_model: MultiChannelNMConfig | None = None

    mmse_count: int = 1
    is_supervised: bool = False

    # Optional fields
    optimizer: OptimizerConfig = OptimizerConfig()
    """Optimizer to use, defined in SupportedOptimizer."""

    lr_scheduler: LrSchedulerConfig = LrSchedulerConfig()

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
        # microsplit
        if self.algorithm == SupportedAlgorithm.MICROSPLIT:
            if self.loss.loss_type != SupportedLoss.MICROSPLIT:
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `microsplit`."
                )

            # Remind users to attach a noise model when using denoiSplit
            if self.loss.denoisplit_weight > 0 and self.noise_model is None:
                warnings.warn(
                    "denoisplit_weight > 0 but no noise_model is provided in the "
                    "configuration. A noise model is required for denoiSplit training. "
                    "Train one with NoiseModelTrainer, then either pass the result to "
                    "create_microsplit_configuration("
                    "noise_model_config=trainer.get_config()) "
                    "or call VAEModule.set_noise_model() before training.",
                    UserWarning,
                    stacklevel=2,
                )
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
        # Check consistency between model.predict_logvar and loss.predict_logvar
        model_predicts_logvar = self.model.predict_logvar
        loss_predicts_logvar = self.loss.predict_logvar

        assert model_predicts_logvar == loss_predicts_logvar, (
            f"Model `predict_logvar` ({model_predicts_logvar}) "
            f"must match loss `predict_logvar` ({loss_predicts_logvar})."
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

    @classmethod
    def get_compatible_algorithms(cls) -> list[str]:
        """Get the list of compatible algorithms.

        Returns
        -------
        list of str
            List of compatible algorithms.
        """
        return ["hdn", "microsplit"]
