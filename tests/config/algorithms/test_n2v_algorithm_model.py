import pytest

from careamics.config.algorithms import N2VAlgorithm


def test_n_channels_n2v():
    """Check that an error is raised if n2v has different number of channels in
    input and output."""
    model = {
        "architecture": "UNet",
        "in_channels": 1,
        "num_classes": 2,
        "n2v2": False,
    }
    loss = "n2v"

    with pytest.raises(ValueError):
        N2VAlgorithm(algorithm="n2v", loss=loss, model=model)


def test_channels(minimum_algorithm_n2v: dict):
    """Check that error is thrown if the number of channels are different."""
    minimum_algorithm_n2v["model"] = {
        "architecture": "UNet",
        "in_channels": 2,
        "num_classes": 2,
        "n2v2": False,
    }
    N2VAlgorithm(**minimum_algorithm_n2v)

    minimum_algorithm_n2v["model"]["num_classes"] = 3
    with pytest.raises(ValueError):
        N2VAlgorithm(**minimum_algorithm_n2v)


def test_no_final_activation(minimum_algorithm_n2v: dict):
    """Check that an error is raised if the model has a final activation."""
    minimum_algorithm_n2v["model"] = {
        "architecture": "UNet",
        "final_activation": "ReLU",
    }
    with pytest.raises(ValueError):
        N2VAlgorithm(**minimum_algorithm_n2v)
