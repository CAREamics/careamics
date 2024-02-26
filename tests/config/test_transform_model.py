import pytest

from careamics.config.support import SupportedTransform
from careamics.config.transform_model import TransformModel


@pytest.mark.parametrize(
    "name, parameters",
    [
        (SupportedTransform.FLIP, {}),
        (SupportedTransform.RANDOM_ROTATE90, {}),
        (SupportedTransform.NORMALIZE, {"mean": 1.0, "std": 1.0}),
        (SupportedTransform.MANIPULATE_N2V, {}),
    ],
)
def test_official_transforms(name, parameters):
    """Test that officially supported transforms are accepted."""
    TransformModel(name=name, parameters=parameters)


def test_nonexisting_transform():
    """Test that non-existing transforms are not accepted."""
    with pytest.raises(ValueError):
        TransformModel(name="OptimusPrime")


def test_filtering_unknown_parameters():
    """Test that unknown parameters are filtered out."""
    parameters = {"some_param": 42, "p": 1.0}

    # create transform model
    transform = TransformModel(name=SupportedTransform.FLIP, parameters=parameters)

    # check parameters
    assert transform.parameters == {"p": 1.0}


def test_missing_parameters():
    """Test that missing parameters trigger an error."""
    with pytest.raises(ValueError):
        TransformModel(name="RandomCrop", parameters={})
