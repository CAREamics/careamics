from __future__ import annotations

from pprint import pformat
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch.nn import Module

from .register_model import get_custom_model


class CustomParametersModel(BaseModel):
    """A Pydantic model that allows any parameter."""

    model_config = ConfigDict(extra="allow")


class CustomModel(BaseModel):
    """Custom model configuration.

    This Pydantic model allows storing parameters for a custom model. In order for the
    model to be valid, the specific model needs to be registered using the
    `register_model` decorator, and its name correctly passed to this model
    configuration (see Examples).

    Attributes
    ----------
    architecture : Literal["Custom"]
        Discriminator for the custom model, must be set to "Custom".
    name : str
        Name of the custom model.
    parameters : CustomParametersModel
        Parameters of the custom model.

    Raises
    ------
    ValueError
        If the custom model `name` is unknown.
    ValueError
        If the custom model is not a torch Module subclass.
    ValueError
        If the custom model parameters are not valid.

    Examples
    --------
    >>> from torch import nn, ones
    >>> from careamics.config import CustomModel, register_model
    >>>
    >>> @register_model(name="linear")
    >>> class LinearModel(nn.Module):
    >>>    def __init__(self, in_features, out_features, *args, **kwargs):
    >>>        super().__init__()
    >>>        self.in_features = in_features
    >>>        self.out_features = out_features
    >>>        self.weight = nn.Parameter(ones(in_features, out_features))
    >>>        self.bias = nn.Parameter(ones(out_features))
    >>>    def forward(self, input):
    >>>        return (input @ self.weight) + self.bias
    >>>
    >>> config_dict = {
    >>>     "architecture": "custom",
    >>>     "name": "linear",
    >>>     "parameters": {
    >>>         "in_features": 10,
    >>>         "out_features": 5,
    >>>     },
    >>> }
    >>> config = CustomModel(**config_dict)
    >>> print(config)
    """

    # pydantic model config
    model_config = ConfigDict(validate_assignment=True)

    # discriminator used for choosing the pydantic model in Model
    architecture: Literal["Custom"]

    # name of the custom model
    name: str

    # parameters
    parameters: CustomParametersModel

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
                f'Retrieved class {model} with name "{value}" is not a '
                f"torch.nn.Module subclass."
            )

        return value

    @model_validator(mode="after")
    def check_parameters(self: CustomModel) -> CustomModel:
        """Validate model by instantiating the model with the parameters.

        Returns
        -------
        CustomModel
            The validated model.
        """
        # instantiate model
        try:
            get_custom_model(self.name)(**self.parameters.model_dump())
        except Exception as e:
            raise ValueError(
                f"error while passing parameters to the model: {e}. Verify that all "
                f"mandatory parameters are provided, and that either the model accepts "
                f"*args and **kwargs, or that no additional parameter is provided."
            ) from None

        return self

    def __str__(self) -> str:
        """Pretty string representing the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
