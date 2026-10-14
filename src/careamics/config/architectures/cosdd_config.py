"""COSDD Pydantic model."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from .architecture_config import ArchitectureConfig


class COSDDConfig(ArchitectureConfig):
    """Pydantic model for the COSDD networks.

    COSDD is made of a Ladder VAE encoding the signal into a latent code, an
    autoregressive decoder modelling the (row-)correlated noise, a signal decoder
    mapping the latent code to image space and an optional direct denoiser.

    The defaults follow the reference implementation
    (https://github.com/krulllab/COSDD).
    """

    # pydantic model config
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    architecture: Literal["COSDD"]
    """Name of the architecture."""

    input_shape: tuple[int, ...] = Field(default=(256, 256), validate_default=True)
    """Spatial shape of the training patches, (Z)YX. It determines the number of
    downsampling steps along each axis and the size of the top prior."""

    in_channels: int = Field(default=1, ge=1)
    """Number of channels of the images. Channels are denoised jointly, but the noise
    is assumed independent across channels."""

    s_code_channels: int = Field(default=64, ge=2)
    """Number of feature channels in the latent code describing the clean signal.
    Half of this value is used as the number of latent channels in each VAE layer."""

    n_layers: int = Field(default=8, ge=2)
    """Number of levels in the Ladder VAE and in the direct denoiser."""

    scale_initialisation: bool = False
    """Whether to divide the output of the residual blocks by sqrt(channels). Can
    stabilise training of larger networks, see VDVAE (Child 2020)."""

    noise_direction: Literal["x", "y", "z", "none"] = "x"
    """Axis along which the noise is correlated. `none` assumes pixel-independent
    noise."""

    n_gaussians: int = Field(default=3, ge=1)
    """Number of components of the Gaussian mixture model describing the noise. If
    noise is reproduced in the output, increase this value. If training fails,
    reduce it."""

    use_direct_denoiser: bool = True
    """Whether to train a direct denoiser predicting the average of the signal
    samples. It increases training time but makes prediction deterministic and
    fast."""

    direct_denoiser_loss: Literal["MSE", "L1"] = "MSE"
    """Loss of the direct denoiser. `MSE` makes it predict the mean of the signal
    samples (MMSE), `L1` their coordinate-wise median."""

    checkpointed: bool = True
    """Whether to use activation checkpointing, trading speed for GPU memory."""

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Validate the input shape.

        Parameters
        ----------
        input_shape : tuple of int
            Spatial shape of the training patches.

        Returns
        -------
        tuple of int
            Validated input shape.

        Raises
        ------
        ValueError
            If the input shape is not 2D or 3D, or if any dimension is smaller
            than 2.
        """
        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"Input shape must be 2D (YX) or 3D (ZYX), got {input_shape}."
            )
        if any(s < 2 for s in input_shape):
            raise ValueError(
                f"All input shape dimensions must be at least 2, got {input_shape}."
            )
        return input_shape

    @model_validator(mode="after")
    def validate_noise_direction(self: Self) -> Self:
        """Validate that the noise direction is compatible with the dimensions.

        Returns
        -------
        Self
            Validated model.

        Raises
        ------
        ValueError
            If the noise direction is `z` with a 2D model.
        """
        if self.noise_direction == "z" and not self.is_3D():
            raise ValueError("Noise direction `z` requires a 3D model (ZYX input).")
        return self

    @property
    def conv_dims(self) -> int:
        """Number of spatial dimensions of the convolutions.

        Returns
        -------
        int
            2 or 3.
        """
        return len(self.input_shape)

    @property
    def z_dims(self) -> list[int]:
        """Number of latent channels of each Ladder VAE layer.

        Returns
        -------
        list of int
            Latent channels, one entry per layer.
        """
        return [self.s_code_channels // 2] * self.n_layers

    def is_3D(self) -> bool:
        """Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        return len(self.input_shape) == 3

    def get_num_input_channels(self) -> int:
        """Get the number of input channels.

        Returns
        -------
        int
            Number of input channels.
        """
        return self.in_channels

    def get_num_output_channels(self) -> int:
        """Get the number of output channels.

        COSDD denoises its input, so it has as many output as input channels.

        Returns
        -------
        int
            Number of output channels.
        """
        return self.in_channels

    def uses_batch_norm(self) -> bool:
        """Return whether the model uses batch normalization.

        Returns
        -------
        bool
            Whether the model uses batch normalization.
        """
        return True
