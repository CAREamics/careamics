from careamics.config.transformations import NormalizeModel


def test_setting_image_means_std():
    """Test that we can set the image_means and std values."""
    model = NormalizeModel(name="Normalize", image_means=[0.5], image_stds=[0.5])
    assert model.image_means == [0.5]
    assert model.image_stds == [0.5]

    model.image_means = [0.6]
    model.image_stds = [0.6]
    assert model.image_means == [0.6]
    assert model.image_stds == [0.6]

    model = NormalizeModel(
        name="Normalize",
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
