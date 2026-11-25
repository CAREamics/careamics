"""Configuration for HDN."""

from typing import Self

from pydantic import ValidationInfo, field_validator, model_validator

from careamics.config.algorithms import HDNAlgorithm
from careamics.config.lightning.training_configuration import TrainingConfig
from careamics.config.validators.training_validators import (
    warn_supervised_checkpointing_params,
)

from .configuration import Configuration


class HDNConfiguration(Configuration):
    """HDN-specific configuration.

    Overrides the UNet-specific validators of `Configuration`, which are not
    applicable to LVAE models.
    """

    algorithm_config: HDNAlgorithm

    @model_validator(mode="after")
    def validate_norm_against_channels(self: Self) -> Self:
        """Validate that normalization sizes match the LVAE channels.

        Returns
        -------
        Self
            Validated configuration.
        """
        n_in = (
            len(self.data_config.channels)
            if self.data_config.channels is not None
            else 1
        )
        n_out = self.algorithm_config.model.output_channels
        self.data_config.normalization.validate_size(n_in, n_out)
        return self

    @field_validator("training_config", mode="after")
    @classmethod
    def warn_checkpointing_params(
        cls, training_config: TrainingConfig, info: ValidationInfo
    ) -> TrainingConfig:
        """Warn about checkpointing parameters unsuitable for HDN.

        Parameters
        ----------
        training_config : TrainingConfig
            Training configuration to validate.
        info : pydantic.ValidationInfo
            Validation context for the field.

        Returns
        -------
        TrainingConfig
            Validated training configuration.
        """
        # TODO: default training factory saves best ckpt but HDN is self-supervised
        warn_supervised_checkpointing_params(training_config)
        return training_config
