""""N2V Algorithm configuration."""

from typing import Literal

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

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

    @model_validator(mode="after")
    def algorithm_cross_validation(self: Self) -> Self:
        """Validate the algorithm model for N2V.

        Returns
        -------
        Self
            The validated model.
        """
        if self.model.in_channels != self.model.num_classes:
            raise ValueError(
                "N2V requires the same number of input and output channels. Make "
                "sure that `in_channels` and `num_classes` are the same."
            )

        if self.model.n2v2 is True:
            self.n2v_masking.strategy = "median"

        return self
