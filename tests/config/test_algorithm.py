from pathlib import Path

import pytest

from n2v.config.algorithm import Algorithm


def test_algorithm(test_config):
    """Test that we can instantiate a config with a valid algorithm."""
    algorithm_config = test_config["algorithm"]
    _ = Algorithm(**algorithm_config)


def test_wrong_loss_value(test_config):
    """Test that we cannot instantiate a config with wrong loss value."""
    algorithm_config = test_config["algorithm"]

    # wrong entry
    algorithm_config["loss"] = ["notn2v"]

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


def test_loss_value(test_config):
    """Test that we can instantiate a config with a single loss or a list of
    losses."""
    algorithm_config = test_config["algorithm"]

    # list
    algorithm_config["loss"] = ["n2v", "pn2v"]
    _ = Algorithm(**algorithm_config)

    # single value
    algorithm_config["loss"] = "n2v"
    _ = Algorithm(**algorithm_config)


def test_wrong_model_value(test_config):
    """Test that we cannot instantiate a config with wrong loss value."""

    algorithm_config = test_config["algorithm"]
    algorithm_config["model"] = "wrongmodel"

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


def test_wrong_manipulator_value(test_config):
    """Test that we cannot instantiate a config with wrong loss value."""

    algorithm_config = test_config["algorithm"]
    algorithm_config["pixel_manipulation"] = "notn2v"

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


@pytest.mark.parametrize("num_masked_pixels", [-10, 0, 1025])
def test_wrong_num_masked_pixels_value(test_config, num_masked_pixels):
    """Test that we cannot instantiate a config with wrong number of masked
    pixels."""

    algorithm_config = test_config["algorithm"]
    algorithm_config["num_masked_pixels"] = num_masked_pixels

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)


def test_trained_model_path(tmpdir, test_config):
    """Test that we can instantiate a config without a trained model and that
    the model validation works."""
    key = "trained_model"
    algorithm_config = test_config["algorithm"]

    # check that key is absent
    assert key not in algorithm_config.keys()

    # instantiate model
    my_algo = Algorithm(**algorithm_config)
    assert key not in my_algo.dict().keys()

    # create a non-valid path
    path = Path(tmpdir, "mytrainedmodel.pth")
    algorithm_config[key] = str(path)
    assert not path.exists()

    # check that it fails instantiation
    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)

    # create a valid path without the extension
    path = Path(tmpdir, "mytrainedmodel")
    algorithm_config[key] = str(path)
    path.touch()
    assert path.exists()

    # check that it fails instantiation
    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)

    # finally create a valid path
    path = Path(tmpdir, "mytrainedmodel.pth")
    algorithm_config[key] = str(path)
    path.touch()
    assert path.exists()

    # create a config with a valid path to a trained model
    algorithm_config[key] = str(path)
    my_other_algo = Algorithm(**algorithm_config)
    assert my_other_algo.trained_model == path
