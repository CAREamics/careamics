"""Architectures supported by CAREamics."""

from careamics.utils import BaseEnum


class SupportedArchitecture(str, BaseEnum):
    """Supported architectures.

    # TODO add details, in particular where to find the API for the models

    - UNet: classical UNet compatible with N2V2
    - VAE: variational Autoencoder
    - Custom: custom model registered with `@register_model` decorator
    """

    UNET = "UNet"
    VAE = "VAE"
    CUSTOM = (
        "Custom"  # TODO all the others tags are small letters, except the architect
    )
