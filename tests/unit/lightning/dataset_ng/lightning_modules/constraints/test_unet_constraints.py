import itertools
from contextlib import nullcontext

import pytest

from careamics.config.architectures import UNetConfig
from careamics.lightning.dataset_ng.lightning_modules.constraints import (
    UNetConstraints,
)


def _compatible_shapes(depth: int):
    """Shapes compatible with a UNet of the given depth."""
    return [i * 2**depth for i in range(1, 4)]


def _incompatible_shapes(depth: int):
    """Shapes incompatible with a UNet of the given depth."""
    return [i * 2**depth + 1 for i in range(1, 4)] + [
        i * 2**depth + 2 for i in range(1, 4)
    ]


@pytest.mark.parametrize(
    "x_shape, z_shape, depth, expected_error",
    list(
        itertools.product(
            _compatible_shapes(depth=2),
            [0] + _compatible_shapes(depth=2),
            [2],
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            _compatible_shapes(depth=3),
            [0] + _compatible_shapes(depth=3),
            [3],
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            _incompatible_shapes(depth=2),
            [0] + _compatible_shapes(depth=2),
            [2],
            [pytest.raises(ValueError)],
        )
    )
    + list(
        itertools.product(
            _incompatible_shapes(depth=3),
            [0] + _compatible_shapes(depth=3),
            [3],
            [pytest.raises(ValueError)],
        )
    )
    + list(
        itertools.product(
            _compatible_shapes(depth=2),
            _incompatible_shapes(depth=2),
            [2],
            [pytest.raises(ValueError)],
        )
    )
    + list(
        itertools.product(
            _compatible_shapes(depth=3),
            _incompatible_shapes(depth=3),
            [3],
            [pytest.raises(ValueError)],
        )
    ),
)
def test_validate_input_shape(x_shape, z_shape, depth, expected_error):
    cfg = UNetConfig(
        architecture="UNet",
        depth=depth,
        conv_dims=3 if z_shape > 0 else 2,
    )
    constraints = UNetConstraints(cfg)
    input_shape = (x_shape, x_shape) if z_shape == 0 else (z_shape, x_shape, x_shape)

    with expected_error:
        constraints.validate_input_shape(input_shape)


@pytest.mark.parametrize("length", [1, 4])
def test_validate_input_shape_wrong_length(length):
    depth = 2
    shape = (_compatible_shapes(depth=depth)[0],) * length

    cfg = UNetConfig(
        architecture="UNet",
        depth=depth,
        conv_dims=2,
    )
    constraints = UNetConstraints(cfg)

    with pytest.raises(ValueError, match="Spatial input shape"):
        constraints.validate_input_shape(shape)
