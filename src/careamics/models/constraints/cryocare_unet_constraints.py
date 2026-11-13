"""CryoCARE U-Net model constraints."""

from collections.abc import Sequence

from careamics.config.architectures import UNetConfig

from .unet_constraints import UNetConstraints


class CryoCAREUNetConstraints(UNetConstraints):
    """CryoCARE constraints for a three-dimensional, single-channel U-Net.

    Parameters
    ----------
    model_config : UNetConfig
        The cryoCARE U-Net configuration from which to derive divisibility constraints.
    """

    def __init__(self, model_config: UNetConfig) -> None:
        """Initialize cryoCARE U-Net constraints.

        Parameters
        ----------
        model_config : UNetConfig
            The cryoCARE U-Net configuration from which to derive divisibility
            constraints.
        """
        super().__init__(model_config)

    def validate_input_channels(self, n_channels: int) -> None:
        """Validate that the input contains only a singleton channel dimension.

        Parameters
        ----------
        n_channels : int
            Number of canonical input channels.

        Raises
        ------
        ValueError
            If the input contains anything other than one channel.
        """
        if n_channels != 1:
            raise ValueError(
                "CryoCARE input data must not contain channels. Expected one "
                f"channel, got {n_channels}."
            )

    def validate_target_channels(self, n_channels: int) -> None:
        """Validate that the target contains only a singleton channel dimension.

        Parameters
        ----------
        n_channels : int
            Number of canonical target channels.

        Raises
        ------
        ValueError
            If the target contains anything other than one channel.
        """
        if n_channels != 1:
            raise ValueError(
                "CryoCARE target data must not contain channels. Expected one "
                f"channel, got {n_channels}."
            )

    def validate_spatial_shape(self, input_shape: Sequence[int]) -> None:
        """Validate that the input has three spatial dimensions.

        Parameters
        ----------
        input_shape : Sequence[int]
            Spatial shape of the input tensor.

        Raises
        ------
        ValueError
            If the input is not three-dimensional or is incompatible with the U-Net
            depth.
        """
        if len(input_shape) != 3:
            raise ValueError(
                "CryoCARE input data must have three spatial dimensions (ZYX), "
                f"got shape {tuple(input_shape)}."
            )

        super().validate_spatial_shape(input_shape)
