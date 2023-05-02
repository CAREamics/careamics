import pytest

from n2v.config.algorithm import Algorithm


def test_algorithm_wrong_loss_value(test_config):
    """Test that we cannot instantiate a config with wrong loss value."""

    algorithm_config = test_config["algorithm"]
    algorithm_config["loss"] = ["notn2v"]

    with pytest.raises(ValueError):
        Algorithm(**algorithm_config)
