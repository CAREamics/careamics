"""Configuration for N2V."""

from typing import Self

import numpy as np
from pydantic import model_validator

from careamics.config.algorithms import N2VAlgorithm
from careamics.config.data.patching_strategies import WholePatchingConfig

from .configuration import Configuration


class N2VConfiguration(Configuration):
    """N2V-specific configuration."""

    algorithm_config: N2VAlgorithm

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

    @model_validator(mode="after")
    def monitor_training_when_no_validation(self: Self) -> Self:
        """
        Validate that training loss is monitored when no validation data is used.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If no validation data is used and the monitored metric is not a training
            metric.
        """
        if self.data_config.mode == "training" and self.data_config.n_val_patches == 0:
            monitored_metric = self.algorithm_config.monitor_metric

            if monitored_metric not in ["train_loss", "train_loss_epoch"]:
                raise ValueError(
                    f"No validation data is used (`n_val_patches=0`), but the monitored"
                    f" metric ({monitored_metric}) is not a training metric. Please set"
                    f" `algorithm_config.monitor_metric` to `train_loss` or "
                    f"`train_loss_epoch` to monitor the training loss during training. "
                    f"Note that the `n_val_patches` parameter is ignored if passing "
                    f"validation data. In which case, keep the default value."
                )

        return self
