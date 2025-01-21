import pytest

from careamics.config.algorithms import N2NAlgorithm


def test_instantiation():
    """Test the instantiation of the N2NAlgorithm class."""
    model = {
        "architecture": "UNet",
    }
    N2NAlgorithm(model=model)


def test_no_n2v2():
    """Check that an error is raised if the model is set for n2v2."""
    model = {
        "architecture": "UNet",
        "n2v2": True,
    }

    with pytest.raises(ValueError):
        N2NAlgorithm(model=model)


def test_no_final_activation(minimum_algorithm_supervised: dict):
    """Check that an error is raised if the model has a final activation."""
    minimum_algorithm_supervised["model"] = {
        "architecture": "UNet",
        "final_activation": "ReLU",
    }
    with pytest.raises(ValueError):
        N2NAlgorithm(**minimum_algorithm_supervised)
