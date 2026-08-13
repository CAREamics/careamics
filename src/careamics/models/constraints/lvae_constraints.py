"""LVAE model constraints."""

from collections.abc import Sequence

from careamics.config.architectures import LVAEConfig


class LVAEConstraints:
    """LVAE model constraints on input tensors spatial shape.

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
        """Whether the given channel size is compatible with the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels is not compatible with the model constraints.
        """
        # the LVAE encoder always takes a single channel, the output channels being
        # either the denoised input (HDN) or the unmixed channels (MicroSplit)
        if n_channels != 1:
            raise ValueError(
                f"Number of channels in input image ({n_channels}) does not match the "
                f"single input channel expected by the LVAE model. Use the `channels` "
                f"parameter to select a single channel."
            )

    def validate_target_channels(self, n_channels: int) -> None:
        """Whether the given channel size is compatible with the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the target tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels is not compatible with the model constraints.
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

        Shape must be of length 2 (YX) or 3 (ZYX). To validate channel dimension, use
        `validate_input_channels` or `validate_target_channels` instead.

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
                f" 3 (ZYX), but got shape {input_shape}."
            )

        dim_label = "ZYX" if len(input_shape) == 3 else "YX"

        # each spatial dimension is downsampled once per hierarchy level by its
        # encoder stride, mirroring `lvae_spatial_shape_valid` on the configuration
        n_levels = len(self.model_config.z_dims)
        strides = self.model_config.encoder_conv_strides
        for i, (dim, stride) in enumerate(zip(input_shape, strides, strict=True)):
            factor = stride**n_levels
            if dim % factor != 0 or dim == 0:
                raise ValueError(
                    f"Input data dimension {dim_label[i]} (size {dim}) is not a "
                    f"multiple of {factor} (encoder stride {stride} to the power of "
                    f"the {n_levels} hierarchy levels). If you are training, adjust "
                    f"`patch_size`. If you are predicting, your input data shape is "
                    f"not compatible, use tiling by passing `tile_size`. If you are "
                    f"already using tiling, adjust `tile_size`."
                )
