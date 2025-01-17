"""N2N Algorithm configuration."""

from typing import Annotated, Literal

from pydantic import AfterValidator

from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_without_final_activation,
    model_without_n2v2,
)

from .unet_algorithm_model import UNetBasedAlgorithm


class N2NAlgorithm(UNetBasedAlgorithm):
    """Noise2Noise Algorithm configuration."""

    algorithm: Literal["n2n"] = "n2n"
    """N2N Algorithm name."""

    loss: Literal["mae", "mse"] = "mae"
    """N2N-compatible loss function."""

    model: Annotated[
        UNetModel,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
    ]
    """UNet without a final activation function and without the `n2v2` modifications."""
