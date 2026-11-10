from contextlib import nullcontext

import pytest

from careamics.config.architectures import UNetConfig
from careamics.models.constraints import CryoCAREUNetConstraints


@pytest.fixture
def constraints() -> CryoCAREUNetConstraints:
    """Return constraints for the default CryoCARE U-Net."""
    return CryoCAREUNetConstraints(
        UNetConfig(
            architecture="UNet",
            conv_dims=3,
            in_channels=1,
            num_classes=1,
            depth=3,
        )
    )


@pytest.mark.parametrize("n_channels", [1])
def test_validate_channels(constraints, n_channels):
    """Accept the singleton canonical input and target channel."""
    with nullcontext():
        constraints.validate_input_channels(n_channels)
        constraints.validate_target_channels(n_channels)


@pytest.mark.parametrize("n_channels", [0, 2])
def test_validate_input_channels_rejects_channels(constraints, n_channels):
    """Reject inputs with no singleton canonical channel."""
    with pytest.raises(ValueError, match="input data must not contain channels"):
        constraints.validate_input_channels(n_channels)


@pytest.mark.parametrize("n_channels", [0, 2])
def test_validate_target_channels_rejects_channels(constraints, n_channels):
    """Reject targets with no singleton canonical channel."""
    with pytest.raises(ValueError, match="target data must not contain channels"):
        constraints.validate_target_channels(n_channels)


def test_validate_spatial_shape_requires_3d(constraints):
    """Reject two-dimensional CryoCARE data."""
    with pytest.raises(ValueError, match="three spatial dimensions"):
        constraints.validate_spatial_shape((64, 64))


def test_validate_spatial_shape_preserves_unet_constraints(constraints):
    """Apply the U-Net divisibility constraint to 3D data."""
    constraints.validate_spatial_shape((64, 64, 64))

    with pytest.raises(ValueError, match="Input data dimension"):
        constraints.validate_spatial_shape((65, 64, 64))
