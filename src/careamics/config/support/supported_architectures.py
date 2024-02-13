from aenum import StrEnum


class SupportedArchitecture(StrEnum):
    """Supported architectures.
    
    # TODO add details, in particular where to find the API for the models

    - UNet: classical UNet compatible with N2V2
    - VAE: variational Autoencoder
    """
    _init_ = 'value __doc__'
    
    UNET = "UNet", "Classical UNet compatible with N2V2."
    VAE = "VAE", "Variational Autoencoder."
    # HVAE?
    # CUSTOM = "Custom" # TODO create mechanism for that
