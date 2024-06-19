import pytest

from careamics.config.transformations import NormalizeModel
from careamics.transforms import Normalize


def test_setting_image_means_std():
    """Test that we can set the image_means and std values."""
    model = NormalizeModel(image_means=[0.5], image_stds=[0.5])
    assert model.image_means == [0.5]
    assert model.image_stds == [0.5]

    model.image_means = [0.6]
    model.image_stds = [0.6]
    assert model.image_means == [0.6]
    assert model.image_stds == [0.6]

    model = NormalizeModel(
        image_means=[0.5],
        image_stds=[0.5],
        target_means=[0.5],
        target_stds=[0.5],
    )
    assert model.image_means == [0.5]
    assert model.image_stds == [0.5]
    assert model.target_means == [0.5]
    assert model.target_stds == [0.5]

    model.image_means = [0.6]
    model.image_stds = [0.6]
    model.target_means = [0.6]
    model.target_stds = [0.6]

    assert model.image_means == [0.6]
    assert model.image_stds == [0.6]
    assert model.target_means == [0.6]
    assert model.target_stds == [0.6]


def test_error_different_length_means_stds():
    """Test that an error is raised if the image_means and stds have different
    lengths."""
    with pytest.raises(ValueError):
        NormalizeModel(image_means=[0.5], image_stds=[0.5, 0.6])

    with pytest.raises(ValueError):
        NormalizeModel(image_means=[0.5, 0.6], image_stds=[0.5])

    with pytest.raises(ValueError):
        NormalizeModel(
            image_means=[0.5],
            image_stds=[0.5],
            target_means=[0.5],
        )

    with pytest.raises(ValueError):
        NormalizeModel(
            image_means=[0.5],
            image_stds=[0.5],
            target_stds=[0.5],
        )

    with pytest.raises(ValueError):
        NormalizeModel(
            image_means=[0.5],
            image_stds=[0.5],
            target_means=[0.5, 0.6],
            target_stds=[0.5],
        )

    with pytest.raises(ValueError):
        NormalizeModel(
            image_means=[0.5],
            image_stds=[0.5],
            target_means=[0.5],
            target_stds=[0.5, 0.6],
        )


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = NormalizeModel(image_means=[0.5], image_stds=[0.5])

    # instantiate transform
    Normalize(**model.model_dump())
