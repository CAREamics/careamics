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

    def validate_input_shape(self, input_shape: Sequence[int]) -> None:
        """Whether the given spatial shape is compatible with the model constraints.

        The input shape should be of size 2 (YX) or 3 (ZYX). Each spatial dimension
        should be a multiple of `2**depth`, where depth is the depth of the UNet model.

        Parameters
        ----------
        input_shape : Sequence[int]
            The shape of the input tensor to validate (length 2 or 3).

        Raises
        ------
        ValueError
            If the input shape is not compatible with the model constraints.
        """
        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"Spatial input shape to model constraints should have length 2 (YX) or"
                f" 3 (ZYX), but got shape {input_shape}."
            )

        # check spatial dims against model depth constraints
        depth = self.model_config.depth
        for i, dim in enumerate(input_shape):
            if dim % (2**depth) != 0 or dim < 2**depth:
                raise ValueError(
                    f"Input data dimension {i} (size {dim}) is not a multiple of "
                    f"2**depth ({2**depth}) or is smaller than 2**depth. Make sure that"
                    f" the input data shape is compatible with the model."
                )
