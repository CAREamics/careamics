from enum import Enum


class SupportedArchitecture(str, Enum):
    """Supported architectures.
    
    # TODO add details, in particular where to find the API for the models

    - UNet: classical UNet compatible with N2V2
    - VAE: variational Autoencoder
    """
    
    UNET = "UNet"
    VAE = "VAE"
    # HVAE?
    # CUSTOM = "Custom" # TODO create mechanism for that
