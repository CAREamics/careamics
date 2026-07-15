"""Configuration for MicroSplit."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.data import MicroSplitDataConfig

from .configuration import Configuration


class MicroSplitConfiguration(Configuration):
    """MicroSplit-specific configuration.

    Overrides the UNet-specific validators of `Configuration`, which are not
    applicable to LVAE models.
    """

    algorithm_config: MicroSplitAlgorithm
    data_config: MicroSplitDataConfig

    # TODO remove once LVAE model constraints have been implemented
    @model_validator(mode="after")
    def validate_patch_against_model(self: Self) -> Self:
        """Skip UNet model-constraint validation for LVAE models.

        Returns
        -------
        Self
            Validated configuration.
        """
        return self

    # TODO remove once LVAE model constraints have been implemented
    @model_validator(mode="after")
    def validate_channels_against_inputs(self: Self) -> Self:
        """Skip UNet channel-constraint validation for LVAE models.

        Returns
        -------
        Self
            Validated configuration.
        """
        return self

    # TODO remove once LVAE model constraints have been implemented
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
