"""Architectures supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedArchitecture(str, BaseEnum):
    """Supported architectures."""

    UNET = "UNet"
    """UNet architecture used with N2V, CARE and Noise2Noise."""

    LVAE = "LVAE"
    """Ladder Variational Autoencoder used for muSplit and denoiSplit."""
