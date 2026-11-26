"""Configuration for PN2V."""

from typing import Self

from pydantic import ValidationInfo, field_validator, model_validator

from careamics.config.algorithms import PN2VAlgorithm
from careamics.config.lightning.training_configuration import TrainingConfig
from careamics.config.validators import (
    check_masked_pixel_probability,
    check_monitor_without_validation,
    check_no_target_axes,
)
from careamics.config.validators.training_validators import (
    warn_supervised_checkpointing_params,
)

from .configuration import Configuration


class PN2VConfiguration(Configuration):
    """PN2V-specific configuration.

    Validators match `N2VConfiguration`: no target axes, at least one masked pixel per
    patch, and a training metric when validation patches are disabled.
    """

    algorithm_config: PN2VAlgorithm

    @model_validator(mode="after")
    def validate_norm_against_channels(self: Self) -> Self:
        """Validate the normalization against the input channels only.

        Overrides `Configuration.validate_norm_against_channels`. The `num_classes`
        outputs of PN2V are prior samples of the input channels, not separate output
        channels, and they are denormalized with the input statistics.

        Returns
        -------
        Self
            Validated configuration.
        """
        n_channels = self.algorithm_config.model.get_num_input_channels()
        self.data_config.normalization.validate_size(n_channels, n_channels)
        return self

    @model_validator(mode="after")
    def target_axes_must_be_none(self: Self) -> Self:
        """Ensure PN2V does not use separate target axes.

        Returns
        -------
        Self
            Validated configuration.
        """
        check_no_target_axes(self.data_config, "PN2V")
        return self

    @model_validator(mode="after")
    def validate_n2v_mask_pixel_perc(self: Self) -> Self:
        """
        Validate that there will always be at least one blind-spot pixel in every patch.

        Returns
        -------
        Self
            Validated configuration.
        """
        check_masked_pixel_probability(
            self.data_config,
            self.algorithm_config.n2v_config.masked_pixel_percentage,
            self.algorithm_config.model.is_3D(),
        )
        return self

    @model_validator(mode="after")
    def monitor_training_when_no_validation(self: Self) -> Self:
        """
        Validate that training loss is monitored when no validation data is used.

        Returns
        -------
        Self
            Validated configuration.
        """
        check_monitor_without_validation(
            self.data_config, self.algorithm_config.monitor_metric
        )
        return self

    @field_validator("training_config", mode="after")
    @classmethod
    def warn_checkpointing_params(
        cls, training_config: TrainingConfig, info: ValidationInfo
    ) -> TrainingConfig:
        """Warn about checkpointing parameters unsuitable for PN2V.

        PN2V ranks its checkpoints by the validation loss.

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
