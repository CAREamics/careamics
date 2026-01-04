"""UNet Pydantic model."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .architecture_config import ArchitectureConfig


# TODO tests activation <-> pydantic model, test the literals!
# TODO annotations for the json schema?
class UNetConfig(ArchitectureConfig):
    """
    Pydantic model for a N2V(2)-compatible UNet.

    Supports 1D, 2D, and 3D convolutions for different data types.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    conv_dims : Literal[1, 2, 3]
        Dimensions (1D, 2D or 3D) of the convolutional layers.
    """

    # pydantic model config
    model_config = ConfigDict(validate_assignment=True)

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["UNet"]
    """Name of the architecture."""

    # parameters
    # validate_defaults allow ignoring default values in the dump if they were not set
    conv_dims: Literal[1, 2, 3] = Field(
        default=2, validate_default=True
    )  # Added 1D support
    """Dimensions (1D, 2D or 3D) of the convolutional layers."""

    num_classes: int = Field(default=1, ge=1, validate_default=True)
    """Number of classes or channels in the model output."""

    in_channels: int = Field(default=1, ge=1, validate_default=True)
    """Number of channels in the input to the model."""

    depth: int = Field(default=2, ge=1, le=10, validate_default=True)
    """Number of levels in the UNet."""

    num_channels_init: int = Field(default=32, ge=8, le=1024, validate_default=True)
    """Number of convolutional filters in the first layer of the UNet."""

    # TODO we are not using this, so why make it a choice?
    final_activation: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU"
    ] = Field(default="None", validate_default=True)
    """Final activation function."""

    n2v2: bool = Field(default=False, validate_default=True)
    """Whether to use N2V2 architecture modifications, with blur pool layers and fewer
    skip connections.
    """

    independent_channels: bool = Field(default=True, validate_default=True)
    """Whether information is processed independently in each channel, used to train
    channels independently."""

    use_batch_norm: bool = Field(default=True, validate_default=True)
    """Whether to use batch normalization in the model."""

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

    @model_validator(mode="after")
    def validate_dimensionality_constraints(self) -> UNetConfig:
        """
        Validate constraints specific to different dimensionalities.

        Returns
        -------
        UNetModel
            Validated model.

        Raises
        ------
        ValueError
            If parameters are incompatible with the specified dimensionality.
        """
        # 1D-specific validations
        if self.conv_dims == 1:
            # For 1D, recommend smaller depth to avoid over-downsampling
            if self.depth > 4:
                import warnings

                warnings.warn(
                    f"Depth of {self.depth} may be too large for 1D data. "
                    f"Consider using depth <= 4 for better performance.",
                    UserWarning,
                    stacklevel=2,
                )

            # N2V2 with 1D may not be optimal
            if self.n2v2:
                import warnings

                warnings.warn(
                    "N2V2 blur-pool layers may not be optimal for 1D data. "
                    "Consider using standard N2V (n2v2=False) for 1D applications.",
                    UserWarning,
                    stacklevel=2,
                )

        # 3D-specific validations (existing)
        elif self.conv_dims == 3:
            if self.depth > 4:
                import warnings

                warnings.warn(
                    f"Depth of {self.depth} may be too large for 3D data. "
                    f"Consider using depth <= 4 to manage memory usage.",
                    UserWarning,
                    stacklevel=2,
                )

        return self

    def set_1D(self, is_1D: bool = True) -> None:
        """
        Set 1D model by setting the `conv_dims` parameters.

        Parameters
        ----------
        is_1D : bool, optional
            Whether the algorithm is 1D or not, by default True.
        """
        if is_1D:
            self.conv_dims = 1
        else:
            # Default back to 2D if not 1D
            self.conv_dims = 2

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

    def is_1D(self) -> bool:
        """
        Return whether the model is 1D or not.

        Returns
        -------
        bool
            Whether the model is 1D or not.
        """
        return self.conv_dims == 1

    def is_2D(self) -> bool:
        """
        Return whether the model is 2D or not.

        Returns
        -------
        bool
            Whether the model is 2D or not.
        """
        return self.conv_dims == 2

    def is_3D(self) -> bool:
        """
        Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        return self.conv_dims == 3
