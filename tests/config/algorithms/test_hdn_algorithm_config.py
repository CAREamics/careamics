# TODO - revisit: tests do not follow the new test organization and style
import pytest

from careamics.config.algorithms import HDNAlgorithm
from careamics.config.architectures import LVAEConfig
from careamics.config.losses.loss_config import HDNLossConfig


def test_instantiation(minimum_algorithm_hdn: dict):
    """Test the instantiation of the HDNAlgorithm class."""
    config = HDNAlgorithm(**minimum_algorithm_hdn)
    assert config.algorithm == "hdn"
    assert config.loss.loss_type == "hdn"
    assert config.optimizer.name == "Adamax"
    assert not config.is_supervised()


def test_default_loss():
    """Test that the loss defaults to the HDN loss."""
    config = HDNAlgorithm(model=LVAEConfig(architecture="LVAE"))
    assert config.loss.loss_type == "hdn"


def test_wrong_algorithm(minimum_algorithm_hdn: dict):
    """Test that another algorithm name is rejected."""
    minimum_algorithm_hdn["algorithm"] = "microsplit"
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)


def test_no_multiscale(minimum_algorithm_hdn: dict):
    """Test that multiscale models are rejected."""
    minimum_algorithm_hdn["model"]["multiscale_count"] = 2
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)


def test_single_output_channel(minimum_algorithm_hdn: dict):
    """Test that more than one output channel is rejected."""
    minimum_algorithm_hdn["model"]["output_channels"] = 2
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)


def test_predict_logvar_mismatch(minimum_algorithm_hdn: dict):
    """Test that model and loss `predict_logvar` must match."""
    minimum_algorithm_hdn["model"]["predict_logvar"] = False
    minimum_algorithm_hdn["loss"] = HDNLossConfig(predict_logvar=True)
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)


def test_mmse_count_lower_bound(minimum_algorithm_hdn: dict):
    """Test that `mmse_count` must be at least 1."""
    minimum_algorithm_hdn["mmse_count"] = 0
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)
