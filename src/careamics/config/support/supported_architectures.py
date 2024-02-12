from aenum import StrEnum

class SupportedArchitecture(StrEnum):
    
    UNET = "UNet"
    VAE = "VAE"
    # CUSTOM = "Custom" # TODO create mechanism for that
