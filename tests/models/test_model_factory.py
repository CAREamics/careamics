import pytest

from careamics.config.architectures import UNetModel
from careamics.models.model_factory import model_factory


# TODO generalize to other architecture
def test_model_registry_unet():
    model_config = {
        "architecture": "UNet",
    }

    # instantiate model
    model_factory(UNetModel(**model_config))
