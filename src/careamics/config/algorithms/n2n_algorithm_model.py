"""N2N Algorithm configuration."""

from typing import Literal

from pydantic import field_validator

from careamics.config.architectures import UNetModel

from .unet_algorithm_model import UNetBasedAlgorithm


class N2NAlgorithm(UNetBasedAlgorithm):
    """N2N Algorithm configuration."""

    algorithm: Literal["n2n"] = "n2n"
    """N2N Algorithm name."""

    loss: Literal["mae", "mse"] = "mae"
    """N2N-compatible loss function."""

    @classmethod
    @field_validator("model")
    def model_without_n2v2(cls, value: UNetModel) -> UNetModel:
        """Validate that the model does not have the n2v2 attribute.

        Parameters
        ----------
        value : UNetModel
            Model to validate.

        Returns
        -------
        UNetModel
            The validated model.
        """
        if value.n2v2:
            raise ValueError(
                "The N2N algorithm does not support the `n2v2` attribute. "
                "Set it to `False`."
            )

        return value
