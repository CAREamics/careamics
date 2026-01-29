"""Configuration for UNet segmentation."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms.seg_unet_algorithm_config import SegAlgorithm

from .ng_configuration import NGConfiguration


class SegConfiguration(NGConfiguration):
    """Configuration for segmentation tasks."""

    algorithm_config: SegAlgorithm
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    # TODO technically this should be in NGConfiguration, but that would require
    # splitting it between segmentation and denoising data configs
    @model_validator(mode="after")
    def no_channel_extraction(self: Self) -> Self:
        """Validate that the `channel` parameter is set to `None` for segmentation.

        Returns
        -------
        Self
            Validated configuration.

        Raises
        ------
        ValueError
            If the `channel` parameter is not `None`.
        """
        if self.data_config.channels is not None:
            raise ValueError(
                "The `channel` parameter must be set to `None` for segmentation "
                "tasks, as all channels are used for prediction."
            )
        return self
