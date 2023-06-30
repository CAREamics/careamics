from enum import Enum
from typing import List

from pydantic import BaseModel, Field, validator


# python 3.11: https://docs.python.org/3/library/enum.html
class Losses(str, Enum):
    """Available loss functions.

    Currently supported:
        - n2v: Noise2Void loss.
    """

    N2V = "n2v"


class Models(str, Enum):
    """Available models.

    Currently supported:
        - UNet: U-Net model.
    """

    UNET = "UNet"


class MaskingStrategies(str, Enum):
    """Available masking strategies.

    Currently supported:
    - default: default masking strategy of Noise2Void.
    """

    DEFAULT = "default"


class ModelParameters(BaseModel):
    """Model parameters.

    The number of filters (base) must be a power of two.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_filters_base : int
        Number of filters of the first level of the network, should be a power
        of 2 (default 96).
    """

    depth: int = Field(ge=1, le=10)
    num_filters_base: int

    @validator("num_filters_base")
    def num_filter_base_must_be_power_of_two(cls, num_filters):
        """Validate that num_filter_base is a power of two."""
        if not num_filters & (num_filters - 1) == 0:
            raise ValueError("Number of filters (base) must be a power of two.")

        return num_filters


class Algorithm(BaseModel):
    """Algorithm configuration used to configure the model and the loss.

    The minimum algorithm configuration is composed of the following fields:
        - loss:
            List of losses to use, currently only supports n2v.
        - model:
            Model to use, currently only supports UNet.
        - is_3D:
            Whether to use a 3D model or not, this should be coherent with the
            data configuration (axes).

    Other optional fields are:
        - masking_strategy:
            Masking strategy to use, currently only supports default masking.
            # TODO explain default masking
        - masked_pixel_percentage:
            Percentage of pixels to be masked in each patch.
        - model_parameters:
            Model parameters, see ModelParameters for more details.

    Attributes
    ----------
    loss : List[Losses]
        List of losses to use, currently only supports n2v.
    model : Models
        Model to use, currently only supports UNet.
    is_3D : bool
        Whether to use a 3D model or not.
    masking_strategy : MaskingStrategies
        Masking strategy to use, currently only supports default masking.
    masked_pixel_percentage : float
        Percentage of pixels to be masked in each patch.
    model_parameters : ModelParameters
        Model parameters, see ModelParameters for more details.
    """

    # Mandatory fields
    loss: List[Losses]
    model: Models
    is_3D: bool

    # Optional fields, define a default value
    masking_strategy: MaskingStrategies = MaskingStrategies.DEFAULT
    masked_pixel_percentage: float = Field(default=0.2, ge=0.1, le=20)
    model_parameters: ModelParameters = ModelParameters()

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable
