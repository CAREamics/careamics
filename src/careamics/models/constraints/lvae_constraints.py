"""LVAE model constraints."""

from collections.abc import Sequence

from careamics.config.architectures import LVAEConfig


class LVAEConstraints:
    """LVAE model constraints on input/output tensors spatial shape and channels.

    Parameters
    ----------
    model_config : LVAEConfig
        The LVAE model configuration from which to derive constraints.
    """

    def __init__(self, model_config: LVAEConfig) -> None:
        """Constructor.

        Parameters
        ----------
        model_config : LVAEConfig
            The LVAE model configuration from which to derive constraints.
        """
        self.model_config = model_config

    def validate_input_channels(self, n_channels: int) -> None:
        """Validate the number of input channels against the model constraints.

        No-op for LVAE, as the current implementation is not compatible with
        multi-channel.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor to validate.
        """

    def validate_target_channels(self, n_channels: int) -> None:
        """Validate the number of target channels against the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the target tensor to validate.

        Raises
        ------
        ValueError
            If the number of target channels does not match the model's number of
            output channels.
        """
        if n_channels != self.model_config.output_channels:
            raise ValueError(
                f"Number of channels in target image ({n_channels}) does not match the "
                f"number of output channels expected by the model configuration "
                f"({self.model_config.output_channels}). Adjust the number of output "
                f"channels in the configuration to match your data."
            )

    def validate_spatial_shape(self, input_shape: Sequence[int]) -> None:
        """Whether the given spatial shape is compatible with the model constraints.

        Each spatial dimension is downsampled once per hierarchy level (there are
        ``len(z_dims)`` levels) by its convolutional stride, so it must be divisible by
        ``stride ** len(z_dims)``. Dimensions with a stride of 1 are unconstrained.
        Shape must be of length 2 (YX) or 3 (ZYX). To validate
        the channel dimension, use `validate_input_channels` or
        `validate_target_channels` instead.

        Parameters
        ----------
        input_shape : Sequence[int]
            The spatial shape of the input tensor to validate (length 2 or 3).

        Raises
        ------
        ValueError
            If the spatial shape is not compatible with the model constraints.
        """
        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"Spatial input shape to model constraints should have length 2 (YX) or"
                f" 3 (ZYX), but got shape {tuple(input_shape)}."
            )

        strides = self.model_config.encoder_conv_strides
        if len(input_shape) != len(strides):
            raise ValueError(
                f"Spatial input shape {tuple(input_shape)} (length {len(input_shape)}) "
                f"does not match the model's encoder convolution strides "
                f"{list(strides)} (length {len(strides)}). The data and model "
                f"dimensionality (2D/3D) must agree."
            )

        dim_label = "ZYX" if len(input_shape) == 3 else "YX"
        n_levels = len(self.model_config.z_dims)

        for i, (dim, stride) in enumerate(zip(input_shape, strides, strict=True)):
            factor = stride**n_levels
            if dim == 0 or dim % factor != 0:
                raise ValueError(
                    f"Input data dimension {dim_label[i]} (size {dim}) is not a "
                    f"multiple of {factor} (encoder stride {stride} to the power of "
                    f"the number of hierarchy levels {n_levels}). If you are "
                    f"training, adjust `patch_size`. If you are predicting, use "
                    f"tiling by passing `tile_size`, or adjust it if already tiling."
                )
