"""Configuration for semantic segmentation."""

from typing import Self

from pydantic import model_validator

from careamics.config.algorithms import SegAlgorithm

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
            If `data_config.normalization.skip_target` is not `True`.
        """
        norm = self.data_config.normalization
        if norm.name != "none" and not norm.skip_target:
            raise ValueError(
                f"Normalization {norm} must have parameter `skip_target` set to `False`"
                f" for segmentation tasks."
            )
        return self
