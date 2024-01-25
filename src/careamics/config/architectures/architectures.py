from enum import Enum

class Architecture(str, Enum):
    
    UNET = "UNet"
    VAE = "VAE"