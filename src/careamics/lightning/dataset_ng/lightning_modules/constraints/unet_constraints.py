"""UNet model constraints."""

from collections.abc import Sequence

from careamics.config.architectures import UNetConfig


class UNetConstraints:
    """UNet model constraints on input tensors spatial shape.

    Parameters
    ----------
    model_config : UNetConfig
        The UNet model configuration from which to derive constraints.
    """

    def __init__(self, model_config: UNetConfig) -> None:
        """Constructor.

        Parameters
        ----------
        model_config : UNetConfig
            The UNet model configuration from which to derive constraints.
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
        # validate input channels
        if n_channels != self.model_config.get_num_input_channels():
            raise ValueError(
                f"Number of channels in input image ({n_channels}) does not match the "
                f"number of input channels expected by the model configuration "
                f"({self.model_config.get_num_input_channels()}). Use the `channels` "
                f"parameter to specify a subset of channels, or adjust the number of "
                f"input channels in the configuration to match your data."
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
        # validate target channels
        if n_channels != self.model_config.get_num_output_channels():
            raise ValueError(
                f"Number of channels in target image ({n_channels}) does not match the "
                f"number of output channels expected by the model configuration "
                f"({self.model_config.get_num_output_channels()}). Adjust the number of"
                f" output channels in the configuration to match your data."
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

        # check spatial dims against model depth constraints
        depth = self.model_config.depth
        for i, dim in enumerate(input_shape):
            if dim % (2**depth) != 0 or dim < 2**depth:
                raise ValueError(
                    f"Input data dimension {dim_label[i]} (size {dim}) is not a "
                    f"multiple of {2**depth} (2 to the power of the model depth). If "
                    f"you are training, adjust `patch_size`. If you are predicting,"
                    f" your input data shape is not compatible, use tiling by passing "
                    f"`tile_size`. If you are already using tiling, adjust `tile_size`."
                )
