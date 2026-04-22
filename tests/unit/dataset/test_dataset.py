import itertools
from contextlib import nullcontext

import numpy as np
import pytest

from careamics.config.data.data_config import DataConfig
from careamics.dataset.dataset import (
    _adjust_shape_for_channels,
    _patch_size_within_data_shapes,
    _shapes_all_equal,
    _validate_shapes_against_mode,
    _validate_shapes_against_model,
)
from careamics.dataset.image_stack import InMemoryImageStack
from careamics.dataset.patch_extractor import PatchExtractor
from tests.utils import data_config_dict_testing

# ------------------------ Test utilities --------------------------


def _make_extractor(*shapes: tuple[int, ...]) -> PatchExtractor:
    """Build a PatchExtractor from SC(Z)YX shapes."""
    stacks = [
        InMemoryImageStack(source="array", data=np.zeros(shape)) for shape in shapes
    ]
    return PatchExtractor(stacks)


class _AlwaysValid:
    """Model constraints that never raise errors.

    See careamics.models.model_constraints.ModelConstraints protocol.
    """

    def validate_input_channels(self, n_channels: int) -> None:
        pass

    def validate_target_channels(self, n_channels: int) -> None:
        pass

    def validate_spatial_shape(self, shape) -> None:
        pass


class _ChannelThreshold:
    """Model constraints that rejects a specific channel count.

    See careamics.models.model_constraints.ModelConstraints protocol.
    """

    def __init__(self, input_size: int, target_size: int) -> None:
        self.input_size = input_size
        self.target_size = target_size

    def validate_input_channels(self, n_channels: int) -> None:
        if n_channels != self.input_size:
            raise ValueError("Invalid channels")

    def validate_target_channels(self, n_channels: int) -> None:
        if n_channels != self.target_size:
            raise ValueError("Invalid channels")

    def validate_spatial_shape(self, shape) -> None:
        pass


class _SpatialThreshold:
    """Model constraints that rejects spatial shapes.

    See careamics.models.model_constraints.ModelConstraints protocol.
    """

    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    def validate_input_channels(self, n_channels: int) -> None:
        pass

    def validate_target_channels(self, n_channels: int) -> None:
        pass

    def validate_spatial_shape(self, shape) -> None:
        for s in shape:
            if s < self.input_size:
                raise ValueError("Invalid spatial shape")


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize(
    "shape, channels, expected_shape",
    [
        ((1, 1, 32, 32), None, (1, 1, 32, 32)),
        ((1, 1, 32, 32), [0], (1, 1, 32, 32)),
        ((5, 4, 32, 32), None, (5, 4, 32, 32)),
        ((5, 4, 32, 32), [1], (5, 1, 32, 32)),
        ((5, 4, 32, 32), [1, 3], (5, 2, 32, 32)),
    ],
)
def test_adjust_shape_for_channels(shape, channels, expected_shape):
    adjusted_shape = _adjust_shape_for_channels(shape, channels)
    assert adjusted_shape == expected_shape


@pytest.mark.parametrize(
    "data_shapes, patch_size, expected",
    [
        # patch fits
        ([(1, 1, 64, 64)], (32, 32), True),
        ([(1, 1, 64, 64), (1, 1, 32, 32)], (32, 32), True),
        ([(1, 1, 32, 32, 32)], (16, 16, 16), True),
        # patch exactly same shape
        ([(1, 1, 32, 32)], (32, 32), True),
        # patch larger than one image
        ([(1, 1, 16, 16)], (32, 32), False),
        ([(1, 1, 64, 64), (1, 1, 16, 16)], (32, 32), False),
        ([(1, 1, 8, 32, 32)], (16, 16, 16), False),
    ],
)
def test_patch_size_within_data_shapes(data_shapes, patch_size, expected):
    assert _patch_size_within_data_shapes(data_shapes, patch_size) == expected


@pytest.mark.parametrize(
    "data_shapes, expected",
    [
        ([(1, 1, 32, 32)], True),
        ([(1, 1, 32, 32), (1, 1, 32, 32)], True),
        ([(1, 1, 32, 32), (1, 1, 64, 64)], False),
    ],
)
def test_shapes_all_equal(data_shapes, expected):
    assert _shapes_all_equal(data_shapes) == expected


# We could avoid specifying patching and use the default from data_config_dict_testing,
# but should the default change, we may break the tests.
@pytest.mark.parametrize(
    "mode, patching, size, constraints, exp_error",
    [
        # valid cases
        ("predicting", "whole", 32, _AlwaysValid(), nullcontext(0)),
        ("predicting", "whole", 128, _SpatialThreshold(64), nullcontext(0)),
        # non-whole are always valid due to patching
        ("training", "stratified", 32, _SpatialThreshold(64), nullcontext(0)),
        # invalid cases
        (
            "predicting",
            "whole",
            32,
            _SpatialThreshold(64),
            pytest.raises(ValueError, match="Invalid spatial shape"),
        ),
    ],
)
def test_validate_model_spatial(mode, patching, size, constraints, exp_error):
    config_dict = data_config_dict_testing(mode=mode, patching=patching)
    config = DataConfig(**config_dict)

    data_shapes = [(1, 1, 128, 128), (1, 1, size, 128)]
    with exp_error:
        _validate_shapes_against_model(config, constraints, data_shapes)


@pytest.mark.parametrize(
    "mode, patching, channels, constraints, exp_error",
    [
        # always valid with channel subset
        ("predicting", "whole", [0, 1], _ChannelThreshold(1, 1), nullcontext(0)),
        ("training", "stratified", [0, 1], _ChannelThreshold(1, 1), nullcontext(0)),
        # valid cases
        ("predicting", "whole", None, _ChannelThreshold(3, 5), nullcontext(0)),
        ("training", "stratified", None, _ChannelThreshold(3, 5), nullcontext(0)),
    ]
    # invalid cases
    + list(
        itertools.product(
            ["predicting"],
            ["whole"],
            [None],
            [_ChannelThreshold(2, 5), _ChannelThreshold(3, 4)],
            [pytest.raises(ValueError, match="Invalid channels")],
        )
    )
    + list(
        itertools.product(
            ["training"],
            ["stratified"],
            [None],
            [_ChannelThreshold(2, 5), _ChannelThreshold(3, 4)],
            [pytest.raises(ValueError, match="Invalid channels")],
        )
    ),
)
def test_validate_model_channels(mode, patching, channels, constraints, exp_error):
    config_dict = data_config_dict_testing(
        mode=mode, patching=patching, axes="CYX", channels=channels
    )
    config = DataConfig(**config_dict)

    data_shapes = [(1, 3, 128, 128), (1, 3, 128, 128)]
    target_shapes = [(1, 5, 128, 128), (1, 5, 128, 128)]
    with exp_error:
        _validate_shapes_against_model(config, constraints, data_shapes, target_shapes)


@pytest.mark.parametrize(
    "mode, patching, batch_size, patch_size, size, exp_error",
    [
        # valid cases
        ("training", "stratified", 2, (64, 64), 128, nullcontext(0)),
        ("predicting", "whole", 2, (64, 64), 128, nullcontext(0)),
        # invalid patching
        (
            "training",
            "stratified",
            2,
            (64, 64),
            32,
            pytest.raises(ValueError, match="Not all images"),
        ),
        # invalid batch size for whole patching with different shapes
        (
            "predicting",
            "whole",
            2,
            (64, 64),
            64,
            pytest.raises(ValueError, match="For prediction without tiling"),
        ),
    ],
)
def test_validate_shapes_against_mode(
    mode, patching, batch_size, patch_size, size, exp_error
):
    config_dict = data_config_dict_testing(
        mode=mode, patching=patching, patch_size=patch_size, batch_size=batch_size
    )
    config = DataConfig(**config_dict)

    data_shapes = [(1, 1, 128, 128), (1, 1, 128, size)]
    with exp_error:
        _validate_shapes_against_mode(config, data_shapes)
