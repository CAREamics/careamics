""""N2V Algorithm configuration."""

from typing import Annotated, Literal

from pydantic import ConfigDict, AfterValidator

from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_matching_in_out_channels,
    model_without_final_activation,
)

from careamics.config.transformations import N2VManipulateModel

from .unet_algorithm_model import UNetBasedAlgorithm


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    model_config = ConfigDict(validate_assignment=True)

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

    n2v_masking: N2VManipulateModel = N2VManipulateModel()

    model: Annotated[
        UNetModel,
        AfterValidator(model_matching_in_out_channels),
        AfterValidator(model_without_final_activation),
    ]
