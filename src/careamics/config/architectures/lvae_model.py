"""LVAE Pydantic model."""

from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .architecture_model import ArchitectureModel


# TODO: it is quite confusing to call this LVAEModel, as it is basically a config
class LVAEModel(ArchitectureModel):
    """LVAE model."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    architecture: Literal["LVAE"]
    """Name of the architecture."""

    input_shape: list[int] = Field(default=[64, 64], validate_default=True)
    """Shape of the input patch (C, Z, Y, X) or (C, Y, X) if the data is 2D."""

    encoder_conv_strides: list = Field(default=[2, 2], validate_default=True)

    # TODO make this per hierarchy step ?
    decoder_conv_strides: list = Field(default=[2, 2], validate_default=True)
    """Dimensions (2D or 3D) of the convolutional layers."""

    multiscale_count: int = Field(default=1)
    # TODO there should be a check for multiscale_count in dataset !!

    # 1 - off, len(z_dims) + 1 # TODO Consider starting from 0
    z_dims: list = Field(default=[128, 128, 128, 128])
    output_channels: int = Field(default=1, ge=1)
    encoder_n_filters: int = Field(default=64, ge=8, le=1024)
    decoder_n_filters: int = Field(default=64, ge=8, le=1024)
    encoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    decoder_dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = Field(
        default="ELU",
    )

    predict_logvar: Literal[None, "pixelwise"] = None
    analytical_kl: bool = Field(default=False)

    @model_validator(mode="after")
    def validate_conv_strides(self: Self) -> Self:
        """
        Validate the convolutional strides.

        Returns
        -------
        list
            Validated strides.

        Raises
        ------
        ValueError
            If the number of strides is not 2.
        """
        if len(self.encoder_conv_strides) < 2 or len(self.encoder_conv_strides) > 3:
            raise ValueError(
                f"Strides must be 2 or 3 (got {len(self.encoder_conv_strides)})."
            )

        if len(self.decoder_conv_strides) < 2 or len(self.decoder_conv_strides) > 3:
            raise ValueError(
                f"Strides must be 2 or 3 (got {len(self.decoder_conv_strides)})."
            )

        # adding 1 to encoder strides for the number of input channels
        if len(self.input_shape) != len(self.encoder_conv_strides):
            raise ValueError(
                f"Input dimensions must be equal to the number of encoder conv strides"
                f" (got {len(self.input_shape)} and {len(self.encoder_conv_strides)})."
            )

        if len(self.encoder_conv_strides) < len(self.decoder_conv_strides):
            raise ValueError(
                f"Decoder can't be 3D when encoder is 2D (got"
                f" {len(self.encoder_conv_strides)} and"
                f"{len(self.decoder_conv_strides)})."
            )

        if any(s < 1 for s in self.encoder_conv_strides) or any(
            s < 1 for s in self.decoder_conv_strides
        ):
            raise ValueError(
                f"All strides must be greater or equal to 1"
                f"(got {self.encoder_conv_strides} and {self.decoder_conv_strides})."
            )
        # TODO: validate max stride size ?
        return self

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
        return input_shape

    @field_validator("encoder_n_filters")
    @classmethod
    def validate_encoder_even(cls, encoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        encoder_n_filters : int
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
        if encoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {encoder_n_filters})."
            )

        return encoder_n_filters

    @field_validator("decoder_n_filters")
    @classmethod
    def validate_decoder_even(cls, decoder_n_filters: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        decoder_n_filters : int
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
        if decoder_n_filters % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {decoder_n_filters})."
            )

        return decoder_n_filters

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

    @model_validator(mode="after")
    def validate_multiscale_count(self: Self) -> Self:
        """
        Validate the multiscale count.

        Returns
        -------
        Self
            The validated model.
        """
        if self.multiscale_count < 1 or self.multiscale_count > len(self.z_dims) + 1:
            raise ValueError(
                f"Multiscale count must be 1 for LC off or less or equal to the number"
                f" of Z dims + 1 (got {self.multiscale_count} and {len(self.z_dims)})."
            )
        return self

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
