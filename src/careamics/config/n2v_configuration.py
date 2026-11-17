"""Configuration for N2V."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms import N2VAlgorithm
from careamics.config.validators import (
    check_masked_pixel_probability,
    check_monitor_without_validation,
    check_no_target_axes,
)

from .configuration import Configuration


class N2VConfiguration(Configuration):
    """N2V-specific configuration."""

    algorithm_config: N2VAlgorithm
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    @model_validator(mode="after")
    def target_axes_must_be_none(self: Self) -> Self:
        """Ensure N2V does not use separate target axes.

        Returns
        -------
        Self
            Validated configuration.
        """
        check_no_target_axes(self.data_config, "N2V")
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
