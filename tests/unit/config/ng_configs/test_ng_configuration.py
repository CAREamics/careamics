"""Unit tests for the NGConfiguration Pydantic model."""

from contextlib import nullcontext

import pytest

from careamics.config.ng_configs import N2VConfiguration, NGConfiguration
from careamics.config.ng_factories.ng_config_discriminator import validate_ng_config
from tests.utils import unet_ng_config_dict_testing

ALGORITHMS = ["care", "n2n", "n2v"]
ALGORITHMS_CONFIGS = [NGConfiguration, NGConfiguration, N2VConfiguration]

# ------------------------ Test utilities --------------------------


def test_default_unet_config():
    """Test that the default NGConfiguration can be created."""
    unet_config_dict = unet_ng_config_dict_testing()
    validate_ng_config(unet_config_dict)


@pytest.mark.parametrize(
    "algorithm, config_class", list(zip(ALGORITHMS, ALGORITHMS_CONFIGS, strict=True))
)
def test_unet_configs(algorithm, config_class):
    """Test that an NGConfiguration can be created for each UNet-based algorithm."""
    unet_config_dict = unet_ng_config_dict_testing(algorithm=algorithm)
    cfg = validate_ng_config(unet_config_dict)
    assert isinstance(cfg, config_class)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize(
    "n_input, n_output, quantiles, expected_exception",
    [
        # valid case
        (2, 2, [0.15, 0.10], nullcontext()),
        # fail if n_input != n_output
        (
            1,
            2,
            [0.15],
            pytest.raises(ValueError, match="Quantile normalization per channel"),
        ),
        # fail if n_input != n_quantiles
        (2, 2, [0.15], pytest.raises(ValueError, match="Number of quantiles")),
    ],
)
def test_validation_quantile_norm(n_input, n_output, quantiles, expected_exception):
    """Test the quantile normalization validation against channels."""
    cfg_dict = unet_ng_config_dict_testing(
        algorithm="care",  # so that we can set different channels in and out
        n_channels_in=n_input,
        n_channels_out=n_output,
        data_kwargs={
            "normalization": {
                "name": "quantile",
                "per_channel": True,
                "lower_quantiles": quantiles,
                "upper_quantiles": [q + 0.2 for q in quantiles],
            }
        },
    )
    with expected_exception:
        validate_ng_config(cfg_dict)
