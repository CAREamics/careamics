"""COSDD model constraints."""

from collections.abc import Sequence

from careamics.config.architectures import COSDDConfig


class COSDDConstraints:
    """COSDD model constraints on input/output tensors spatial shape and channels.

    The COSDD networks pad their input internally, so any spatial shape matching the
    model dimensionality is accepted.

    Parameters
    ----------
    model_config : COSDDConfig
        The COSDD model configuration from which to derive constraints.
    """

    def __init__(self, model_config: COSDDConfig) -> None:
        """Constructor.

        Parameters
        ----------
        model_config : COSDDConfig
            The COSDD model configuration from which to derive constraints.
        """
        self.model_config = model_config

    def validate_input_channels(self, n_channels: int) -> None:
        """Validate the number of input channels against the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels does not match the model input channels.
        """
        if n_channels != self.model_config.in_channels:
            raise ValueError(
                f"Number of channels in the input image ({n_channels}) does not match "
                f"the number of input channels expected by the model configuration "
                f"({self.model_config.in_channels}). Use the `channels` parameter to "
                f"select a subset of channels, or adjust the number of input channels "
                f"in the configuration."
            )

    def validate_target_channels(self, n_channels: int) -> None:
        """Validate the number of target channels against the model constraints.

        Parameters
        ----------
        n_channels : int
            The number of channels in the target tensor to validate.

        Raises
        ------
        ValueError
            If the number of channels does not match the model output channels.
        """
        if n_channels != self.model_config.in_channels:
            raise ValueError(
                f"Number of channels in the target image ({n_channels}) does not "
                f"match the number of output channels expected by the model "
                f"configuration ({self.model_config.in_channels})."
            )

    def validate_spatial_shape(self, input_shape: Sequence[int]) -> None:
        """Validate the spatial shape against the model dimensionality.

        Parameters
        ----------
        input_shape : Sequence of int
            The spatial shape of the input tensor to validate (length 2 or 3).

        Raises
        ------
        ValueError
            If the length of the spatial shape does not match the model dimensions.
        """
        if len(input_shape) != self.model_config.conv_dims:
            raise ValueError(
                f"Spatial input shape {tuple(input_shape)} does not match the "
                f"model dimensionality ({self.model_config.conv_dims}D)."
            )
