"""LVAE Pydantic model."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from .architecture_config import ArchitectureConfig


# TODO: it is quite confusing to call this LVAEModel, as it is basically a config
class LVAEConfig(ArchitectureConfig):
    """LVAE model."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    architecture: Literal["LVAE"]

    input_shape: tuple[int, ...] = Field(default=(64, 64), validate_default=True)
    """Shape of the input patch (Z, Y, X) or (Y, X) if the data is 2D."""
    encoder_conv_strides: list = Field(default=[2, 2], validate_default=True)

    # TODO make this per hierarchy step ?
    decoder_conv_strides: list = Field(default=[2, 2], validate_default=True)
    """Dimensions (2D or 3D) of the convolutional layers."""

    multiscale_count: int = Field(default=1)

    # 1 - off, len(z_dims) + 1 # TODO Consider starting from 0
    z_dims: list = Field(default=[128, 128, 128, 128])
    output_channels: int = Field(default=1, ge=1)
    n_filters: int = Field(default=64, ge=8, le=1024)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    decoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    encoder_blocks_per_layer: int = Field(default=1, ge=1)
    """Number of residual blocks per encoder layer."""
    decoder_blocks_per_layer: int = Field(default=1, ge=1)
    """Number of residual blocks per decoder layer."""
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = Field(
        default="ELU",
    )

    predict_logvar: bool = True
    """Whether to predict log-variance (pixelwise uncertainty)."""

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, input_shape: list) -> list:
        """
        Validate the input shape.

        Parameters
        ----------
        input_shape : list
            Shape of the input patch.

        Returns
        -------
        list
            Validated input shape.

        Raises
        ------
        ValueError
            If the number of dimensions is not 3 or 4.
        """
        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(
                f"Number of input dimensions must be 2 for 2D data 3 for 3D"
                f"(got {len(input_shape)})."
            )

        if any(s < 1 for s in input_shape):
            raise ValueError(
                f"Input shape must be greater than 1 in all dimensions"
                f"(got {input_shape})."
            )

        if any(s < 64 for s in input_shape[-2:]):
            raise ValueError(
                f"Input shape must be greater or equal to 64 in XY dimensions"
                f"(got {input_shape})."
            )

        return input_shape

    @field_validator("n_filters")
    @classmethod
    def validate_n_filters_even(cls, n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        n_filters : int
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
        if n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {n_filters})."
            )

        return n_filters

    @field_validator("z_dims")
    def validate_z_dims(cls, z_dims: tuple) -> tuple:
        """
        Validate the z_dims.

        Parameters
        ----------
        z_dims : tuple
            Tuple of z dimensions.

        Returns
        -------
        tuple
            Validated z dimensions.

        Raises
        ------
        ValueError
            If the number of z dimensions is not 4.
        """
        if len(z_dims) < 2:
            raise ValueError(
                f"Number of z dimensions must be at least 2 (got {len(z_dims)})."
            )

        return z_dims

    def is_3D(self) -> bool:
        """
        Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        return len(self.input_shape) == 3

    def uses_batch_norm(self) -> bool:
        """
        Return whether the model uses batch normalization.

        LVAE models do not use batch normalization.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False
