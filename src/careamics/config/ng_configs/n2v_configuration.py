"""Configuration for N2V."""

from typing import Self

import numpy as np
from pydantic import model_validator

from careamics.config.algorithms import N2VAlgorithm
from careamics.config.data.patching_strategies import RandomPatchingConfig

from .ng_configuration import NGConfiguration


class N2VConfiguration(NGConfiguration):
    """N2V-specific configuration."""

    algorithm_config: N2VAlgorithm

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
            assert isinstance(self.data_config.patching, RandomPatchingConfig)

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
