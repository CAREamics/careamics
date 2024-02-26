from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator
)
from torch.nn import Module

from .register_model import get_custom_model

class CustomModel(BaseModel):

    # pydantic model config
    model_config = ConfigDict(
        validate_assignment=True
    )

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["Custom"]

    # name of the custom model
    name: str

    # parameters
    parameters: dict = {}


    @field_validator("name")
    @classmethod
    def custom_model_is_known(cls, value: str) -> str:
        """Check whether the custom model is known.

        Parameters
        ----------
        value : str
            Name of the custom model as registered using the `@register_model` 
            decorator.
        """
        # delegate error to get_custom_model
        model = get_custom_model(value)

        # check if it is a torch Module subclass
        if not issubclass(model, Module):
            raise ValueError(
                f"Retrieved class {model} with name \"{value}\" is not a "
                f"torch.nn.Module subclass."
            )

        return value