from types import SimpleNamespace

import pytest

from careamics.config.algorithms import (
    CAREAlgorithm,
    HDNAlgorithm,
    MicroSplitAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
    SegAlgorithm,
)
from careamics.config.architectures import LVAEConfig, UNetConfig
from careamics.models.constraints import (
    LVAEConstraints,
    SegUNetConstraints,
    UNetConstraints,
    get_model_constraints,
)


@pytest.mark.parametrize("algorithm", [N2NAlgorithm, N2VAlgorithm, CAREAlgorithm])
def test_get_model_constraints_unet(algorithm):
    """UNet configs are dispatched to `UNetConstraints`."""
    constraints = get_model_constraints(
        algorithm(model=UNetConfig(architecture="UNet"))
    )
    assert isinstance(constraints, UNetConstraints)


def test_get_model_constraints_seg_unet():
    """UNet configs are dispatched to `UNetConstraints`."""
    constraints = get_model_constraints(
        SegAlgorithm(
            model=UNetConfig(
                architecture="UNet",
                num_classes=2,
                independent_channels=False,
            )
        )
    )
    assert isinstance(constraints, SegUNetConstraints)


@pytest.mark.parametrize("algorithm", [HDNAlgorithm, MicroSplitAlgorithm])
def test_get_model_constraints_lvae(algorithm):
    """LVAE configs are dispatched to `LVAEConstraints`."""
    constraints = get_model_constraints(
        algorithm(model=LVAEConfig(architecture="LVAE"))
    )
    assert isinstance(constraints, LVAEConstraints)


def test_get_model_constraints_unsupported_raises():
    """An unsupported architecture raises a `ValueError`."""
    with pytest.raises(ValueError, match="has no corresponding model"):
        get_model_constraints(SimpleNamespace(algorithm="Unknown", model="ViT"))
