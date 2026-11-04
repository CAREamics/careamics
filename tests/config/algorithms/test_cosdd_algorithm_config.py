"""Tests for the COSDD algorithm configuration."""

from careamics.config import COSDDAlgorithm
from careamics.config.factories.config_discriminators import (
    instantiate_algorithm_config,
)


def _algorithm_dict() -> dict:
    return {"algorithm": "cosdd", "model": {"architecture": "COSDD"}}


def test_defaults():
    """Test that the optimization defaults follow the reference implementation."""
    config = COSDDAlgorithm(**_algorithm_dict())

    assert config.optimizer.name == "Adamax"
    assert config.optimizer.parameters == {}
    assert config.lr_scheduler.name == "ReduceLROnPlateau"
    assert config.lr_scheduler.parameters == {"factor": 0.1, "patience": 25}
    assert config.n_grad_batches == 4
    assert not config.is_supervised()


def test_algorithm_information():
    """Test the algorithm name, keywords and references."""
    config = COSDDAlgorithm(**_algorithm_dict())

    assert config.get_algorithm_friendly_name() == "COSDD"
    assert "COSDD" in config.get_algorithm_keywords()
    assert "2D" in config.get_algorithm_keywords()
    assert "Salmon" in config.get_algorithm_references()
    assert len(config.get_algorithm_citations()) == 1
    assert "correlated" in config.get_algorithm_description()


def test_instantiate_algorithm_config():
    """Test that the algorithm discriminator returns a COSDD configuration."""
    config = instantiate_algorithm_config(_algorithm_dict())

    assert isinstance(config, COSDDAlgorithm)
