from careamics.config.architectures import UNetModel
from careamics.models import model_factory
from careamics.models.unet import UNet


# TODO improve tests
def test_model_registry_unet():
    """Test that"""
    model_config = {
        "architecture": "UNet",
    }

    # instantiate model
    model = model_factory(UNetModel(**model_config))
    assert isinstance(model, UNet)
