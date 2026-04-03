"""Unit tests for the NGConfiguration Pydantic model."""

import pytest

from careamics.config.ng_configs import N2VConfiguration, NGConfiguration
from careamics.config.ng_factories.ng_config_discriminator import instantiate_config
from tests.utils import unet_ng_config_dict_testing

ALGORITHMS = ["care", "n2n", "n2v"]
ALGORITHMS_CONFIGS = [NGConfiguration, NGConfiguration, N2VConfiguration]

# ------------------------ Test utilities --------------------------


def test_default_unet_config():
    """Test that the default NGConfiguration can be created."""
    unet_config_dict = unet_ng_config_dict_testing()
    instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "algorithm, config_class", list(zip(ALGORITHMS, ALGORITHMS_CONFIGS, strict=True))
)
def test_unet_configs(algorithm, config_class):
    """Test that an NGConfiguration can be created for each UNet-based algorithm."""
    unet_config_dict = unet_ng_config_dict_testing(algorithm=algorithm)
    cfg = instantiate_config(unet_config_dict)
    assert isinstance(cfg, config_class)


# -------------------------- Unit tests ----------------------------
