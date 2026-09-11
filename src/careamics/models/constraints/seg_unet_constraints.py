"""Segmentation UNet model constraints."""

from careamics.config.architectures import UNetConfig

from .unet_constraints import UNetConstraints


# TODO the SegUnet is now testing the number of channels of the target, meaning
# that it acts more as a task constraint validator than a model one.
class SegUNetConstraints(UNetConstraints):
    """Segmentation UNet model constraints on input and output tensors.

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
        # in segmentation the targets will have n_channels = 1, while the model will
        # output multiple classes as channels
        if n_channels != 1:
            raise ValueError(
                f"Number of channels in target image ({n_channels}) does not match the "
                f"number of channels expected for segmentation targets (1). If your "
                f"targets are one-hot encoded, adjust your data to have a single "
                f"channel with integer class labels."
            )
