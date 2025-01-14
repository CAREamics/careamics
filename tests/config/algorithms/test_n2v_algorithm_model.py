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


def test_comaptiblity_of_number_of_channels(minimum_algorithm_n2v: dict):
    """Check that no error is thrown when instantiating the algorithm with a valid
    number of in and out channels."""
    minimum_algorithm_n2v["model"] = {
        "architecture": "UNet",
        "in_channels": 2,
        "num_classes": 2,
        "n2v2": False,
    }

    N2VAlgorithm(**minimum_algorithm_n2v)
