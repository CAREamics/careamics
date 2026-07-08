import itertools

import pytest

from careamics.config.factories.n2v_factory import (
    _validate_n2v_channel_dim,
    create_advanced_n2v_config,
)
from careamics.config.support import SupportedPixelManipulation, SupportedStructAxis

# TODO structn2v no augs, strucn2v


# --- List of test parameters

VALID_CH_CASES = [
    # axes, channels, n_channels, expected
    ("YX", None, None, 1),
    ("YX", None, 1, 1),
    ("CYX", [0], None, 1),
    ("CYX", [0, 2], None, 2),
    ("CYX", None, 3, 3),
    ("CYX", [0, 2], 2, 2),
]

INVALID_CH_CASES = [
    # axes, channels, n_channels, error_match
    (
        "CYX",
        None,
        None,
        "`n_channels` or `channels` must be specified",
    ),
    (
        "YX",
        [0],
        None,
        "`channels` can only be specified when `axes` includes 'C'",
    ),
    (
        "CYX",
        [],
        None,
        "`channels` cannot not be empty",
    ),
    (
        "YX",
        None,
        2,
        "C is not present in the axes",
    ),
    (
        "CYX",
        [0, 2],
        3,
        "does not match length of `channels`",
    ),
]

USE_N2V2 = [True, False]

STRUCT_AXES = [a.value for a in SupportedStructAxis] + ["none"]

MONITOR_METRIC = ["train_loss", "train_loss_epoch", "val_loss"]

# --- Unit tests


@pytest.mark.parametrize(
    "axes,channels,n_channels,expected",
    VALID_CH_CASES,
)
def test_validate_n2v_channel_dim_valid_cases(
    axes: str,
    channels: list[int] | None,
    n_channels: int | None,
    expected: int,
) -> None:
    assert _validate_n2v_channel_dim(axes, channels, n_channels) == expected


@pytest.mark.parametrize(
    "axes,channels,n_channels,error_match",
    INVALID_CH_CASES,
)
def test_validate_n2v_channel_dim_invalid_cases(
    axes: str,
    channels: list[int] | None,
    n_channels: int | None,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _validate_n2v_channel_dim(axes, channels, n_channels)


@pytest.mark.parametrize(
    "use_n2v2, struct_n2v_axes", list(itertools.product(USE_N2V2, STRUCT_AXES))
)
def test_create_n2v_transform(use_n2v2: bool, struct_n2v_axes: str) -> None:
    """Test that the n2v transform is created with the correct parameters."""
    from careamics.config.factories.n2v_factory import _create_n2v_transform

    roi_size = 13
    masked_pixel_percentage = 0.5
    struct_n2v_span = 7
    seed = 42

    n2v_transform = _create_n2v_transform(
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        use_n2v2=use_n2v2,
        struct_n2v_axes=struct_n2v_axes,
        struct_n2v_span=struct_n2v_span,
        seed=seed,
    )

    assert n2v_transform.roi_size == roi_size
    assert n2v_transform.masked_pixel_percentage == masked_pixel_percentage

    if use_n2v2:
        assert n2v_transform.strategy == SupportedPixelManipulation.MEDIAN.value
    else:
        assert n2v_transform.strategy == SupportedPixelManipulation.UNIFORM.value

    if struct_n2v_axes == "none":
        assert n2v_transform.struct_mask is None
    else:
        assert n2v_transform.struct_mask.axes == struct_n2v_axes
        assert n2v_transform.struct_mask.span == struct_n2v_span

    assert n2v_transform.seed == seed


@pytest.mark.parametrize("monitor_metric", MONITOR_METRIC)
def test_monitor_metric(monitor_metric):
    """Test that the monitor metric is correctly set in the configuration."""
    config = create_advanced_n2v_config(
        experiment_name="test_experiment",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=8,
        monitor_metric=monitor_metric,
    )

    assert config.algorithm_config.monitor_metric == monitor_metric
