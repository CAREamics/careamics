from types import SimpleNamespace

import pytest

from careamics.config.architectures import LVAEConfig, UNetConfig
from careamics.models.constraints import (
    LVAEConstraints,
    UNetConstraints,
    get_model_constraints,
)


def test_get_model_constraints_unet():
    """UNet configs are dispatched to `UNetConstraints`."""
    constraints = get_model_constraints(UNetConfig(architecture="UNet"))
    assert isinstance(constraints, UNetConstraints)


def test_get_model_constraints_lvae():
    """LVAE configs are dispatched to `LVAEConstraints`."""
    constraints = get_model_constraints(LVAEConfig(architecture="LVAE"))
    assert isinstance(constraints, LVAEConstraints)


def test_get_model_constraints_unsupported_raises():
    """An unsupported architecture raises a `ValueError`."""
    with pytest.raises(ValueError, match="is not supported"):
        get_model_constraints(SimpleNamespace(architecture="Unknown"))
