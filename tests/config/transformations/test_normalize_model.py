from careamics.config.transformations import NormalizeModel


def test_setting_mean_std():
    """Test that we can set the mean and std values."""
    model = NormalizeModel(name="Normalize", mean=0.5, std=0.5)
    assert model.mean == 0.5
    assert model.std == 0.5

    model.mean = 0.6
    model.std = 0.6
    assert model.mean == 0.6
    assert model.std == 0.6
