"""UNet Pydantic model."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .architecture_model import ArchitectureModel


# TODO tests activation <-> pydantic model, test the literals!
# TODO annotations for the json schema?
class UNetModel(ArchitectureModel):
    """
    Pydantic model for a N2V(2)-compatible UNet.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    """

    # pydantic model config
    model_config = ConfigDict(validate_assignment=True)

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["UNet"]

    # parameters
    # validate_defaults allow ignoring default values in the dump if they were not set
    conv_dims: Literal[2, 3] = Field(default=2, validate_default=True)
    num_classes: int = Field(default=1, ge=1, validate_default=True)
    in_channels: int = Field(default=1, ge=1, validate_default=True)
    depth: int = Field(default=2, ge=1, le=10, validate_default=True)
    num_channels_init: int = Field(default=32, ge=8, le=1024, validate_default=True)
    final_activation: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU"
    ] = Field(default="None", validate_default=True)
    n2v2: bool = Field(default=False, validate_default=True)
    independent_channels: bool = Field(default=True, validate_default=True)

    @field_validator("num_channels_init")
    @classmethod
    def validate_num_channels_init(cls, num_channels_init: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        num_channels_init : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if num_channels_init % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {num_channels_init})."
            )

        return num_channels_init

    def set_3D(self, is_3D: bool) -> None:
        """
        Set 3D model by setting the `conv_dims` parameters.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        """
        if is_3D:
            self.conv_dims = 3
        else:
            self.conv_dims = 2

    def is_3D(self) -> bool:
        """
        Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        return self.conv_dims == 3
