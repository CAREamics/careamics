from __future__ import annotations

from enum import Enum
from typing import Dict, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from .architectures import UNet

class Architecture(str, Enum):
    """
    Available architectures.

    Currently supported architectures:
        - UNet: U-Net architecture.
    """

    UNET = "UNet"

    @classmethod
    def update_parameters(
        cls, architecture: Union[str, Model], parameters: dict
    ) -> dict:
        """Update parameters for the particular architecture.

        Parameters
        ----------
        architecture : str
            Architecture.

        Returns
        -------
        Updated parameters dictionary

        Raises
        ------
        ValueError
            If the architecture is not supported.
        """
        if architecture == Architecture.UNET:
            UNet(**parameters)
            for k, v in UNet().model_dump().items():
                if k not in parameters:
                    parameters[k] = v
            return parameters
        else:
            raise ValueError(
                f"Unsupported or incorrect architecture {architecture}."
                f"Please refer to the documentation"  # TODO add link to documentation
            )



class Model(BaseModel):
    """
    Available models.

    Currently supported models:
        - UNet: U-Net model.
    """

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    architecture: Architecture
    parameters: Dict = Field(default_factory=dict, validate_default=True)
    is_3D: bool

    @field_validator("parameters")
    def validate_model(cls, data, values: ValidationInfo) -> Dict:
        """Validate model parameters."""
        parameters = Architecture.update_parameters(values.data["architecture"], data)
        return parameters

    def get_conv_dim(self) -> int:
        """
        Get the convolution layers dimension (2D or 3D).

        Returns
        -------
        int
            Dimension (2 or 3).
        """
        return 3 if self.is_3D else 2


    # TODO build_model?