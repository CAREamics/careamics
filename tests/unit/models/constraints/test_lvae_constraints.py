from contextlib import nullcontext

import pytest

from careamics.config.architectures import LVAEConfig
from careamics.models.constraints import LVAEConstraints


def _lvae_config(
    encoder_conv_strides: list[int],
    z_dims: list[int],
    output_channels: int = 1,
) -> LVAEConfig:
    """Build an LVAE config from the values that drive the constraint under test.

    `input_shape` is only needed to construct a valid config (it is not what
    `validate_spatial_shape` checks); its dimensionality follows the strides.
    """
    input_shape = (64, 64) if len(encoder_conv_strides) == 2 else (64, 64, 64)
    return LVAEConfig(
        architecture="LVAE",
        input_shape=input_shape,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=encoder_conv_strides,
        z_dims=z_dims,
        output_channels=output_channels,
    )


@pytest.mark.parametrize(
    "shape, encoder_conv_strides, z_dims, expected_error",
    [
        # stride 2, 2 levels -> factor 2**2 = 4
        ((4, 4), [2, 2], [128, 128], nullcontext()),
        ((64, 64), [2, 2], [128, 128], nullcontext()),
        ((66, 66), [2, 2], [128, 128], pytest.raises(ValueError, match="Input data")),
        # stride 2, 4 levels -> factor 2**4 = 16
        ((64, 64), [2, 2], [128, 128, 128, 128], nullcontext()),
        (
            (72, 72),
            [2, 2],
            [128, 128, 128, 128],
            pytest.raises(ValueError, match="Input data"),
        ),
        # 3D with Z stride 1 -> Z unconstrained (factor 1); XY factor 4
        ((5, 64, 64), [1, 2, 2], [128, 128], nullcontext()),
        (
            (5, 66, 64),
            [1, 2, 2],
            [128, 128],
            pytest.raises(ValueError, match="Input data"),
        ),
        # wrong length (must be 2 or 3)
        ((64,), [2, 2], [128, 128], pytest.raises(ValueError, match="Spatial input")),
        (
            (64, 64, 64, 64),
            [2, 2],
            [128, 128],
            pytest.raises(ValueError, match="Spatial input"),
        ),
        # shape length differs from the model's encoder strides
        (
            (64, 64, 64),
            [2, 2],
            [128, 128],
            pytest.raises(ValueError, match="does not match the model"),
        ),
    ],
)
def test_validate_spatial_shape(shape, encoder_conv_strides, z_dims, expected_error):
    constraints = LVAEConstraints(
        _lvae_config(encoder_conv_strides=encoder_conv_strides, z_dims=z_dims)
    )
    with expected_error:
        constraints.validate_spatial_shape(shape)


@pytest.mark.parametrize(
    "n_channels, expected_error",
    [
        (3, nullcontext()),
        (2, pytest.raises(ValueError, match="target image")),
        (1, pytest.raises(ValueError, match="target image")),
    ],
)
def test_validate_target_channels(n_channels, expected_error):
    constraints = LVAEConstraints(
        _lvae_config(encoder_conv_strides=[2, 2], z_dims=[128, 128], output_channels=3)
    )
    with expected_error:
        constraints.validate_target_channels(n_channels)
