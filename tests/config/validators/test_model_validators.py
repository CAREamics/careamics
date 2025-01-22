import pytest

from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_matching_in_out_channels,
    model_without_final_activation,
    model_without_n2v2,
)


def test_model_without_n2v2():
    """Test the validation of the model without the `n2v2` attribute."""
    model = UNetModel(architecture="UNet", final_activation="None", n2v2=False)
    assert model_without_n2v2(model) == model

    model = UNetModel(architecture="UNet", final_activation="None", n2v2=True)
    with pytest.raises(ValueError):
        model_without_n2v2(model)


def test_model_without_final_activation():
    """Test the validation of the model without the `final_activation` attribute."""
    model = UNetModel(
        architecture="UNet",
        final_activation="None",
    )
    assert model_without_final_activation(model) == model

    model = UNetModel(
        architecture="UNet",
        final_activation="ReLU",
    )
    with pytest.raises(ValueError):
        model_without_final_activation(model)


def test_model_matching_in_out_channels():
    """Test the validation of the model with matching in and out channels."""
    model = UNetModel(
        architecture="UNet",
        in_channels=1,
        num_classes=1,
    )
    assert model_matching_in_out_channels(model) == model

    model = UNetModel(
        architecture="UNet",
        in_channels=1,
        num_classes=2,
    )
    with pytest.raises(ValueError):
        model_matching_in_out_channels(model)
