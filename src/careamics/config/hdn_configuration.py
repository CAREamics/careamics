"""Configuration for HDN."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms import HDNAlgorithm

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
