from enum import Enum


class SupportedArchitecture(str, Enum):
    """Supported architectures.
    
    # TODO add details, in particular where to find the API for the models

    - UNet: classical UNet compatible with N2V2
    - VAE: variational Autoencoder
    - Custom: custom model registered with `@register_model` decorator
    """
    
    UNET = "UNet"
    VAE = "VAE"
    CUSTOM = "Custom"
    # HVAE?
    # CUSTOM = "Custom" # TODO create mechanism for that
