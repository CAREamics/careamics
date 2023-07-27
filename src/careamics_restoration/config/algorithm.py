from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_filter import remove_default_optionals


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
    - default: default masking strategy of Noise2Void (uniform sampling of neighbors).
    - median: median masking strategy of N2V2
    """

    DEFAULT = "default"
    MEDIAN = "median"


class ModelParameters(BaseModel):
    """Model parameters.

    The number of filters (base) must be even and minimum 8.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    """

    model_config = ConfigDict(validate_assignment=True)

    depth: int = Field(default=2, ge=1, le=10)
    num_channels_init: int = 96

    # TODO revisit the constraints on num_channels_init
    @field_validator("num_channels_init")
    def greater_than_eight_and_even(cls, num_channels: int) -> int:
        """Validate that num_channels_init is a power of two (minimum 8)."""
        # if odd
        if num_channels % 2 != 0:
            raise ValueError(
                f"Number of channels (init) must be even (got {num_channels})."
            )

        # if less than 8
        if num_channels < 8:
            raise ValueError(
                f"Number of filters (base) must be at least 8 (got {num_channels})."
            )

        return num_channels


class Algorithm(BaseModel):
    """Algorithm configuration used to configure the model and the loss.

    The minimum algorithm configuration is composed of the following fields:
        - loss:
            Loss to use, currently only supports n2v.
        - model:
            Model to use, currently only supports UNet.
        - is_3D:
            Whether to use a 3D model or not, this should be coherent with the
            data configuration (axes).

    Other optional fields are:
        - masking_strategy:
            Masking strategy to use, currently only supports default masking.
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

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    # Mandatory fields
    loss: Losses
    model: Models
    is_3D: bool

    # Optional fields, define a default value
    masking_strategy: MaskingStrategies = MaskingStrategies.DEFAULT
    masked_pixel_percentage: float = Field(default=0.2, ge=0.1, le=20)
    model_parameters: ModelParameters = ModelParameters()

    def get_conv_dim(self) -> int:
        """Get the convolution layers dimension (2D or 3D).

        Returns
        -------
        int
            Dimension (2 or 3)
        """
        return 3 if self.is_3D else 2

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals is True:
            # remove optional arguments if they are default
            defaults = {
                "masking_strategy": MaskingStrategies.DEFAULT.value,
                "masked_pixel_percentage": 0.2,
                "model_parameters": ModelParameters().model_dump(exclude_none=True),
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
