import pytest

from careamics.config.algorithms import HDNAlgorithm


def test_no_multiscale_hdn(minimum_algorithm_hdn):
    """Test that the multiscale model is not provided for HDN."""
    _ = HDNAlgorithm(**minimum_algorithm_hdn)
    minimum_algorithm_hdn["model"]["multiscale_count"] = 2
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)


def test_target_channel_hdn(minimum_algorithm_hdn):
    """Test that the correct nymber of target channel is provided for HDN."""
    _ = HDNAlgorithm(**minimum_algorithm_hdn)
    minimum_algorithm_hdn["model"]["output_channels"] = 2
    with pytest.raises(ValueError):
        HDNAlgorithm(**minimum_algorithm_hdn)
