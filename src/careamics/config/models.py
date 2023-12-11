from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    validator,
)


class Architecture(str, Enum):
    """
    Available architectures.

    Currently supported architectures:
        - UNet: U-Net architecture.
    """

    UNET = "UNet"

    @classmethod
    def select_architecture(
        cls, architecture: Union[str, Model], parameters: dict
    ) -> None:
        """Validate architecture.

        Parameters
        ----------
        architecture : str
            Architecture.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the architecture is not supported.
        """
        if architecture == Architecture.UNET:
            #TODO validate parameters
            UNet(**parameters)
            return UNet().model_dump() if not parameters else parameters
        else:
            raise ValueError(
                f"Unsupported or incorrect architecture {architecture}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )


class UNet(BaseModel):
    """
    Deep-learning model parameters.

    The number of filters (base) must be even and minimum 8.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    """

    model_config = ConfigDict(
        use_enum_values=True, protected_namespaces=(), validate_assignment=True
    )
    depth: int = Field(default=2, ge=1, le=10)
    num_channels_init: int = Field(default=32, ge=8, le=1024)

    @validator("num_channels_init")
    def validate_num_channels_init(cls, num_channels_init: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        num_channels_init : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if num_channels_init % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {num_channels_init})."
            )

        return num_channels_init


class Model(BaseModel):
    """
    Available models.

    Currently supported models:
        - UNet: U-Net model.
    """

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
    architecture: Architecture
    parameters: Dict = Field(default_factory=dict, validate_default=True)

    @field_validator("parameters")
    def validate_model(cls, data, values: ValidationInfo) -> Dict:
        """Validate model parameters."""
        parameters = Architecture.select_architecture(values.data["architecture"], data)
        return parameters
