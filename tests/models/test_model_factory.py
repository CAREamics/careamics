import pytest

from careamics.config.architectures import (
    CustomModel,
    LVAEModel,
    UNetModel,
)
from careamics.config.support import SupportedArchitecture
from careamics.models import UNet, model_factory


def test_model_registry_unet():
    """Test that"""
    model_config = {
        "architecture": "UNet",
    }

    # instantiate model
    model = model_factory(UNetModel(**model_config))
    assert isinstance(model, UNet)


def test_model_registry_custom(custom_model_parameters):
    """Test that a custom model can be retrieved and instantiated."""
    # instantiate model
    model = model_factory(CustomModel(**custom_model_parameters))
    assert model.in_features == 10
    assert model.out_features == 5


def test_lvae():
    """Test that VAE are currently not supported."""
    model_config = {
        "architecture": SupportedArchitecture.LVAE.value,
    }

    with pytest.raises(NotImplementedError):
        model_factory(LVAEModel(**model_config))
