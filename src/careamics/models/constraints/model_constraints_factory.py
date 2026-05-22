"""Model constraints factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from careamics.config.support import SupportedAlgorithm

if TYPE_CHECKING:
    from careamics.config.factories.config_discriminators import AlgorithmConfig

from .model_constraints import ModelConstraints
from .seg_unet_constraints import SegUNetConstraints
from .unet_constraints import UNetConstraints


def get_model_constraints(algorithm_config: AlgorithmConfig) -> ModelConstraints:
    """Get the model constraints for the given model configuration.

    Parameters
    ----------
    algorithm_config : AlgorithmConfig
        The model configuration.

    Returns
    -------
    ModelConstraints
        The model constraints for the given model configuration.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
    match algorithm_config.algorithm:
        case SupportedAlgorithm.SEG:
            return SegUNetConstraints(algorithm_config.model)
        case SupportedAlgorithm.CARE | SupportedAlgorithm.N2N | SupportedAlgorithm.N2V:
            return UNetConstraints(algorithm_config.model)
        case _:
            raise ValueError(
                f"Algorithm {algorithm_config.algorithm} has no corresponding model"
                f"constraints."
            )
