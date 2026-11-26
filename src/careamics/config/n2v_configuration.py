"""Configuration for N2V."""

from typing import Self

import numpy as np
from pydantic import ValidationInfo, field_validator, model_validator

from careamics.config.algorithms import N2VAlgorithm
from careamics.config.data.patching_strategies import WholePatchingConfig
from careamics.config.lightning.training_configuration import (
    TrainingConfig,
)
from careamics.config.validators.training_validators import (
    warn_self_supervised_checkpointing_params,
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

        Raises
        ------
        ValueError
            If `target_axes` is not `None`.
        """
        if self.data_config.target_axes is not None:
            raise ValueError("N2V does not support non-None `target_axes`.")
        return self

    # TODO note that only patch sizes 4 and 8 may lead to less than 1 expected masked
    # pixel per patch given a minimum masked pixel percentage of 0.05%. This validation
    # could be removed in the future.
    @model_validator(mode="after")
    def validate_n2v_mask_pixel_perc(self: Self) -> Self:
        """
        Validate that there will always be at least one blind-spot pixel in every patch.

        The probability of creating a blind-spot pixel is a function of the chosen
        masked pixel percentage and patch size.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If the probability of masking a pixel within a patch is less than 1 for the
            chosen masked pixel percentage and patch size.
        """
        if self.data_config.mode == "training":
            assert not isinstance(self.data_config.patching, WholePatchingConfig)

            mask_pixel_perc = self.algorithm_config.n2v_config.masked_pixel_percentage
            patch_size = self.data_config.patching.patch_size
            expected_area_per_pixel = 1 / (mask_pixel_perc / 100)

            n_dims = 3 if self.algorithm_config.model.is_3D() else 2
            patch_size_lower_bound = int(
                np.ceil(expected_area_per_pixel ** (1 / n_dims))
            )
            required_patch_size = tuple(
                2 ** int(np.ceil(np.log2(patch_size_lower_bound)))
                for _ in range(n_dims)
            )
            required_mask_pixel_perc = (1 / np.prod(patch_size)) * 100

            if expected_area_per_pixel > np.prod(patch_size):
                raise ValueError(
                    "The probability of creating a blind-spot pixel within a patch is "
                    f"below 1, for a patch size of {patch_size} with a masked pixel "
                    f"percentage of {mask_pixel_perc}%. Either increase the patch size "
                    f"to {required_patch_size} or increase the masked pixel percentage "
                    f"to at least {required_mask_pixel_perc}%."
                )

        return self

    @field_validator("training_config", mode="after")
    @classmethod
    def warn_checkpointing_params(
        cls, training_config: TrainingConfig, info: ValidationInfo
    ) -> TrainingConfig:
        """Warn about checkpointing parameters unsuitable for N2V.

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
        warn_self_supervised_checkpointing_params(training_config)
        return training_config
