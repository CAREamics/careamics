import pytest
from torch import nn, ones

from careamics.config.architectures import CustomModel, get_custom_model, register_model
from careamics.config.support import SupportedArchitecture


@register_model(name="linear")
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(ones(in_features, out_features))
        self.bias = nn.Parameter(ones(out_features))

    def forward(self, input):
        return (input @ self.weight) + self.bias


@register_model(name="not_a_model")
class NotAModel:
    def __init__(self, id):
        self.id = id

    def forward(self, input):
        return input


def test_any_custom_parameters():
    """Test that the custom model can have any fields.

    Note that those fields are validated by instantiating the
    model.
    """
    CustomModel(architecture="Custom", name="linear", in_features=10, out_features=5)


def test_linear_model():
    """Test that the model can be retrieved and instantiated."""
    model = get_custom_model("linear")
    model(in_features=10, out_features=5)


def test_not_a_model():
    """Test that the model can be retrieved and instantiated."""
    model = get_custom_model("not_a_model")
    model(3)


def test_custom_model():
    """Test that the custom model can be instantiated."""
    # prepare model dictionary
    model_dict = {
        "architecture": SupportedArchitecture.CUSTOM.value,
        "name": "linear",
        "in_features": 10,
        "out_features": 5,
    }

    # create Pydantic model
    pydantic_model = CustomModel(**model_dict)

    # instantiate model
    model_class = get_custom_model(pydantic_model.name)
    model = model_class(**pydantic_model.model_dump())

    assert isinstance(model, LinearModel)
    assert model.in_features == 10
    assert model.out_features == 5


def test_custom_model_wrong_class():
    """Test that the Pydantic custom model raises an error if the model is not a
    torch.nn.Module subclass."""
    # prepare model dictionary
    model_dict = {
        "architecture": "Custom",
        "name": "not_a_model",
        "parameters": {"id": 3},
    }

    # create Pydantic model
    with pytest.raises(ValueError):
        CustomModel(**model_dict)


def test_wrong_parameters():
    """Test that the custom model raises an error if the parameters are not valid."""
    # prepare model dictionary
    model_dict = {
        "architecture": "Custom",
        "name": "linear",
        "parameters": {"in_features": 10},
    }

    # create Pydantic model
    with pytest.raises(ValueError):
        CustomModel(**model_dict)
