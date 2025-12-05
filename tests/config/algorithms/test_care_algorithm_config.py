import pytest

from careamics.config.algorithms import CAREAlgorithm


def test_instantiation():
    """Test the instantiation of the CAREAlgorithm class."""
    model = {
        "architecture": "UNet",
    }
    CAREAlgorithm(model=model)


def test_no_n2v2():
    """Check that an error is raised if the model is set for n2v2."""
    model = {
        "architecture": "UNet",
        "n2v2": True,
    }

    with pytest.raises(ValueError):
        CAREAlgorithm(model=model)


def test_no_final_activation(minimum_algorithm_supervised: dict):
    """Check that an error is raised if the model has a final activation."""
    minimum_algorithm_supervised["model"] = {
        "architecture": "UNet",
        "final_activation": "ReLU",
    }
    with pytest.raises(ValueError):
        CAREAlgorithm(**minimum_algorithm_supervised)
