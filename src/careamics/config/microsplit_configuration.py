"""Configuration for MicroSplit."""

from typing import Annotated, Self

from pydantic import AfterValidator, model_validator

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.data import MicroSplitDataConfig
from careamics.config.validators import (
    alpha_ranges_match_output_channels,
    input_shape_matches_patch_size,
    multiscale_counts_match,
    normalization_supported,
)

from .configuration import Configuration


class MicroSplitConfiguration(Configuration):
    """MicroSplit-specific configuration.

    Overrides the UNet-specific validators of `Configuration`, which are not
    applicable to LVAE models, and cross-validates the LVAE model against the
    MicroSplit data configuration.
    """

    algorithm_config: MicroSplitAlgorithm
    data_config: Annotated[
        MicroSplitDataConfig, AfterValidator(normalization_supported)
    ]

    @model_validator(mode="after")
    def validate_patch_against_model(self: Self) -> Self:
        """Validate that the LVAE input shape matches the data patch size.

        Replaces the UNet-specific model-constraint validation, which is not
        applicable to LVAE models.

        Returns
        -------
        Self
            Validated configuration.
        """
        input_shape_matches_patch_size(self.algorithm_config.model, self.data_config)
        return self

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

    @model_validator(mode="after")
    def validate_model_against_data(self: Self) -> Self:
        """Cross-validate the LVAE model against the MicroSplit data configuration.

        Ensures the lateral-context count agrees between model and data, and that
        the channel-mixing ranges match the number of output channels.

        Returns
        -------
        Self
            Validated configuration.
        """
        multiscale_counts_match(self.algorithm_config.model, self.data_config)
        alpha_ranges_match_output_channels(
            self.algorithm_config.model, self.data_config
        )
        return self
