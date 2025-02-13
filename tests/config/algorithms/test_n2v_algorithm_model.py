import pytest

from careamics.config.algorithms import N2VAlgorithm
from careamics.config.support import SupportedPixelManipulation


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


def test_set_n2v_strategy(minimum_algorithm_n2v: dict):
    """Test that the N2V2 can be set."""
    uniform = SupportedPixelManipulation.UNIFORM.value
    median = SupportedPixelManipulation.MEDIAN.value

    # Test default
    cfg = N2VAlgorithm(**minimum_algorithm_n2v)
    assert cfg.n2v_config.strategy == uniform

    # Test setting to n2v2
    cfg.set_n2v2(True)
    assert cfg.n2v_config.strategy == median
    assert cfg.model.n2v2 is True

    # Test setting back to n2v
    cfg.set_n2v2(False)
    assert cfg.n2v_config.strategy == uniform
    assert cfg.model.n2v2 is False
