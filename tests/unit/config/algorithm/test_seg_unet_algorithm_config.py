from contextlib import nullcontext as does_not_raise

import pytest

from careamics.config.algorithms.seg_unet_algorithm_config import (
    _model_with_at_least_2_classes,
    _model_with_dependent_channels,
)
from careamics.config.architectures import UNetConfig


@pytest.mark.parametrize(
    "num_classes, exp_error",
    [(1, pytest.raises(ValueError, match="at least 2 classes")), (2, does_not_raise())],
)
def test_model_with_at_least_2_classes(num_classes, exp_error):
    """Test the validation of a segmentation model output classes."""
    model = UNetConfig(
        architecture="UNet",
        num_classes=num_classes,
        independent_channels=False,
    )
    with exp_error:
        _ = _model_with_at_least_2_classes(model)


@pytest.mark.parametrize(
    "ind_channels, exp_error",
    [
        (True, pytest.raises(ValueError, match="independent_channels")),
        (False, does_not_raise()),
    ],
)
def test_model_with_dependent_channels(ind_channels, exp_error):
    """Test the validation of a segmentation model with dependent channels."""
    model = UNetConfig(
        architecture="UNet",
        num_classes=2,
        independent_channels=ind_channels,
    )
    with exp_error:
        _ = _model_with_dependent_channels(model)
