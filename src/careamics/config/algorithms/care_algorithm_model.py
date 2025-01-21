"""CARE algorithm configuration."""

from typing import Annotated, Literal

from pydantic import AfterValidator

from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_without_final_activation,
    model_without_n2v2,
)

from .unet_algorithm_model import UNetBasedAlgorithm


class CAREAlgorithm(UNetBasedAlgorithm):
    """CARE algorithm configuration.

    Attributes
    ----------
    algorithm : "care"
        CARE Algorithm name.
    loss : {"mae", "mse"}
        CARE-compatible loss function.
    """

    algorithm: Literal["care"] = "care"
    """CARE Algorithm name."""

    loss: Literal["mae", "mse"] = "mae"
    """CARE-compatible loss function."""

    model: Annotated[
        UNetModel,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
    ]
    """UNet without a final activation function and without the `n2v2` modifications."""
