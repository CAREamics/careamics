import pytest

from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from careamics.config.factories.config_discriminators import (
    instantiate_algorithm_config,
)
from tests.utils import unet_algo_dict_testing

# ------------------------ Test utilities --------------------------

ALGORITHMS = ["care", "n2n", "n2v"]

ALGORITHMS_CLASSES = [CAREAlgorithm, N2NAlgorithm, N2VAlgorithm]

# --- Unit tests


def test_default_unet_algorithm_config():
    """Test that the default algorithm can be created."""
    algo_config_dict = unet_algo_dict_testing()
    instantiate_algorithm_config(algo_config_dict)


@pytest.mark.parametrize(
    "algorithm, cfg_class", list(zip(ALGORITHMS, ALGORITHMS_CLASSES, strict=True))
)
def test_unet_algorithm_configs(algorithm, cfg_class):
    """Test that an algorithm config can be created for all UNet-based algorithms."""
    algo_config_dict = unet_algo_dict_testing(algorithm=algorithm)
    cfg = instantiate_algorithm_config(algo_config_dict)
    assert isinstance(cfg, cfg_class)


@pytest.mark.parametrize(
    "algorithm, n_in, n_out",
    [
        # CARE
        ("care", 1, 1),
        ("care", 1, 2),
        ("care", 2, 3),
        # N2N
        ("n2n", 1, 1),
        ("n2n", 1, 2),
        ("n2n", 2, 3),
        # N2V, channels must be equal
        ("n2v", 1, 1),
        ("n2v", 2, 2),
    ],
)
def test_unet_algorithm_config_channels(algorithm, n_in, n_out):
    """Test that an algorithm config can be created for all UNet-based algorithms with
    various channels."""
    algo_config_dict = unet_algo_dict_testing(
        algorithm=algorithm, n_channels_in=n_in, n_channels_out=n_out
    )
    instantiate_algorithm_config(algo_config_dict)
