"""Unit tests for the Configuration Pydantic model."""

import itertools
from contextlib import nullcontext

import pytest

from careamics.config.factories.config_discriminators import instantiate_config
from careamics.config.n2v_configuration import N2VConfiguration
from tests.utils import unet_config_dict_testing

METRICS_TRAIN = ["train_loss", "train_loss_epoch"]
METRCIS_VAL = ["val_loss"]


# -------------------------- Utilities -----------------------------


def n2v_dict_testing(**kwargs):
    """Return a dictionary with default values for testing N2VConfiguration."""
    unet_config_dict = unet_config_dict_testing(algorithm="n2v", **kwargs)
    return unet_config_dict


def _low_percentage_masked_pixels(size: int, dims: int) -> float:
    """Return a masked pixel percentage that would lead to less than 1 expected masked
    pixel per patch for a given patch size and dimensionality.

    Note that N2V manipulate has a minimum masked pixel percentage of 0.05%."""
    return (1 / (size**dims)) * 100 * 0.9


def _high_percentage_masked_pixels(size: int, dims: int) -> float:
    """Return a masked pixel percentage that would lead to at least 1 expected masked
    pixel per patch for a given patch size and dimensionality."""
    return (1 / (size**dims)) * 100 * 1.1


def test_n2v_dict_testing():
    """Test that n2v_dict_testing returns a dictionary that can be used to create an
    N2VConfiguration."""
    unet_config_dict = n2v_dict_testing()
    cfg = instantiate_config(unet_config_dict)
    assert isinstance(cfg, N2VConfiguration)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize(
    "size, dims, percentage, exp_error",
    list(
        itertools.product(
            [8], [2, 3], [_high_percentage_masked_pixels], [nullcontext(0)]
        )
    )
    + list(
        itertools.product(
            [8],
            [2, 3],
            [_low_percentage_masked_pixels],
            [pytest.raises(ValueError, match="The probability of creating")],
        )
    ),
)
def test_number_of_masked_pixels(size, dims, percentage, exp_error):
    """Test that an error is raised when the number of masked pixels is below 1."""
    unet_config_dict = n2v_dict_testing(
        axes="ZYX" if dims == 3 else "YX",
        patch_size=(size,) * dims,
        algo_kwargs={"n2v_config": {"masked_pixel_percentage": percentage(size, dims)}},
    )

    with exp_error:
        instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "metric, exp_error",
    list(itertools.product(METRICS_TRAIN, [nullcontext(0)]))
    + list(
        itertools.product(
            METRCIS_VAL, [pytest.raises(ValueError, match="No validation data")]
        )
    ),
)
def test_monitor_with_no_validation(metric, exp_error):
    """Test that disabling validation but monitoring a validation metric raises an
    error."""
    unet_config_dict = n2v_dict_testing(
        algo_kwargs={"monitor_metric": metric},
        data_kwargs={"n_val_patches": 0},
    )

    with exp_error:
        instantiate_config(unet_config_dict)
