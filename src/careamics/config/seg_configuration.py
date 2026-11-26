"""Configuration for semantic segmentation."""

from typing import Self

from pydantic import ValidationInfo, field_validator, model_validator

from careamics.config.algorithms import SegAlgorithm
from careamics.config.lightning.training_configuration import (
    TrainingConfig,
)
from careamics.config.validators.training_validators import (
    warn_supervised_checkpointing_params,
)

from .configuration import Configuration


class SegConfiguration(Configuration):
    """Segmentation-specific configuration."""

    algorithm_config: SegAlgorithm
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    @model_validator(mode="after")
    def target_normalization_is_skipped(self: Self) -> Self:
        """Ensure that the target are skipped in normalization calculation.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If `data_config.normalization.target_mode` is not `"none"`.
        """
        norm = self.data_config.normalization
        if norm.name != "none" and norm.target_mode != "none":
            raise ValueError(
                f"Normalization {norm} must have parameter `target_mode` set to `none`"
                f" for segmentation tasks."
            )
        return self

    @field_validator("training_config", mode="after")
    @classmethod
    def warn_checkpointing_params(
        cls, training_config: TrainingConfig, info: ValidationInfo
    ) -> TrainingConfig:
        """Warn about checkpointing parameters unsuitable for segmentation.

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
        warn_supervised_checkpointing_params(training_config)
        return training_config
