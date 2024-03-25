import pytest

from careamics.config.support import SupportedTransform
from careamics.config.transformations.transform_model import TransformModel



def test_official_transforms():
    """Test that albumentation transforms are accepted."""
    TransformModel(name="PixelDropout", parameters={
        "dropout_prob": 0.05, 
        "per_channel": True
    })


def test_nonexisting_transform():
    """Test that non-existing transforms are not accepted."""
    with pytest.raises(ValueError):
        TransformModel(name="OptimusPrime")


def test_filtering_unknown_parameters():
    """Test that unknown parameters are filtered out."""
    parameters = {
        "dropout_prob": 0.05,
        "babar_the_elephant": True, 
        "per_channel": True,
    }

    # create transform model
    transform = TransformModel(name="PixelDropout", parameters=parameters)

    # check parameters
    assert transform.parameters.model_dump() == {
        "dropout_prob": 0.05,
        "per_channel": True,
    }


def test_missing_parameters():
    """Test that missing parameters trigger an error."""
    with pytest.raises(ValueError):
        TransformModel(name="RandomCrop", parameters={})
