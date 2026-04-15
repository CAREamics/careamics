"""Architectures supported by CAREamics."""

from enum import StrEnum


class SupportedArchitecture(StrEnum):
    """Supported architectures."""

    UNET = "UNet"
    """UNet architecture used with N2V, CARE and Noise2Noise."""

    LVAE = "LVAE"
    """Ladder Variational Autoencoder used for muSplit and denoiSplit."""
