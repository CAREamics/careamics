from enum import Enum


class Model(str, Enum):
    """
    Available models.

    Currently supported models:
        - UNet: U-Net model.
    """

    UNET = "UNet"
