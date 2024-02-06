from enum import Enum

class SupportedArchitecture(str, Enum):
    
    UNET = "UNet"
    VAE = "VAE"
    CUSTOM = "Custom"
