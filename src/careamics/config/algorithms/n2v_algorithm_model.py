""""N2V Algorithm configuration."""

from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from .unet_algorithm_model import UNetBasedAlgorithm


class N2VAlgorithm(UNetBasedAlgorithm):
    """N2V Algorithm configuration."""

    algorithm: Literal["n2v"] = "n2v"
    """N2V Algorithm name."""

    loss: Literal["n2v"] = "n2v"
    """N2V loss function."""

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

        return self
