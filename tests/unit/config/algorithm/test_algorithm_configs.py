import pytest

from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from careamics.config.ng_factories.ng_config_discriminator import (
    instantiate_algorithm_config,
)
from tests.utils import unet_ng_algo_dict_testing

ALGORITHMS = ["care", "n2n", "n2v"]
ALGORITHMS_CLASSES = [CAREAlgorithm, N2NAlgorithm, N2VAlgorithm]

# ------------------------ Test utilities --------------------------


def test_default_unet_algorithm_config():
    """Test that the default algorithm can be created."""
    algo_config_dict = unet_ng_algo_dict_testing()
    instantiate_algorithm_config(algo_config_dict)


@pytest.mark.parametrize(
    "algorithm, cfg_class", list(zip(ALGORITHMS, ALGORITHMS_CLASSES, strict=True))
)
def test_unet_algorithm_configs(algorithm, cfg_class):
    """Test that an algorithm config can be created for all UNet-based algorithms."""
    algo_config_dict = unet_ng_algo_dict_testing(algorithm=algorithm)
    cfg = instantiate_algorithm_config(algo_config_dict)
    assert isinstance(cfg, cfg_class)


@pytest.mark.parametrize(
    "algorithm, n_in, n_out", list(zip(ALGORITHMS, [1, 2, 3], [1, 2, 3], strict=True))
)
def test_unet_algorithm_config_channels(algorithm, n_in, n_out):
    """Test that an algorithm config can be created for all UNet-based algorithms with
    equal channels."""
    algo_config_dict = unet_ng_algo_dict_testing(
        algorithm=algorithm, n_channels_in=n_in, n_channels_out=n_out
    )
    instantiate_algorithm_config(algo_config_dict)


@pytest.mark.parametrize(
    "algorithm, n_in, n_out", list(zip(["care", "n2n"], [3, 2], [2, 3], strict=True))
)
def test_unet_algorithm_config_diff_channels(algorithm, n_in, n_out):
    """Test that an algorithm config can be created for all UNet-based algorithms with
    different channels."""
    algo_config_dict = unet_ng_algo_dict_testing(
        algorithm=algorithm, n_channels_in=n_in, n_channels_out=n_out
    )
    instantiate_algorithm_config(algo_config_dict)
