import pytest

from careamics.config.factories.care_n2n_factory import _validate_channel_dim

# --- List of test parameters

VALID_CASES = [
    # axes, target_axes, channels, in_channels, out_channels, expected
    # no channels
    ("YX", None, None, None, None, (1, 1, "YX")),
    ("YX", None, None, 1, None, (1, 1, "YX")),
    ("YX", None, None, 1, 1, (1, 1, "YX")),
    ("YX", None, None, None, 1, (1, 1, "YX")),
    ("YX", "ZYX", None, None, None, (1, 1, "ZYX")),
    # with channels
    ("CYX", None, [0], None, None, (1, 1, "CYX")),
    ("CYX", None, [0, 2], None, None, (2, 2, "CYX")),
    ("CYX", None, None, 3, None, (3, 3, "CYX")),
    ("CYX", "CYX", [0, 2], 2, 4, (2, 4, "CYX")),
    ("YX", "CYX", None, 1, 2, (1, 2, "CYX")),
    ("YX", "CYX", None, None, 2, (1, 2, "CYX")),
    ("CYX", "YX", None, 2, 1, (2, 1, "YX")),
    ("CYX", "YX", None, 2, None, (2, 1, "YX")),
    ("CYX", "YXC", None, 2, 1, (2, 1, "YXC")),
    ("CYX", "YXC", None, 1, 2, (1, 2, "YXC")),
    ("CYX", "YXC", None, 2, 3, (2, 3, "YXC")),
    ("CYX", None, None, 2, 3, (2, 3, "CYX")),
]

INVALID_CASES = [
    # axes, target_axes, channels, in_channels, out_channels, error_match
    (
        "CYX",
        None,
        None,
        None,
        None,
        "`n_channels_in` or `channels` must be specified",
    ),
    (
        "YX",
        None,
        [0],
        None,
        None,
        "`channels` can only be specified when `axes` includes 'C'",
    ),
    (
        "CYX",
        None,
        [],
        None,
        None,
        "`channels` cannot not be empty",
    ),
    (
        "YX",
        None,
        None,
        2,
        None,
        "C is not present in the axes",
    ),
    (
        "CYX",
        None,
        [0, 2],
        3,
        None,
        "does not match length of `channels`",
    ),
    (
        "YX",
        None,
        None,
        1,
        2,
        "Number of output channels is greater",
    ),
    (
        "YX",
        "YX",
        None,
        1,
        2,
        "Number of output channels is greater",
    ),
]

# --- Unit tests


@pytest.mark.parametrize(
    "axes, target_axes, channels, n_channels_in, n_channels_out, expected",
    VALID_CASES,
)
def test_validate_channel_dim_valid_cases(
    axes: str,
    target_axes: str | None,
    channels: list[int] | None,
    n_channels_in: int | None,
    n_channels_out: int | None,
    expected: tuple[int, int],
) -> None:
    assert (
        _validate_channel_dim(
            axes=axes,
            target_axes=target_axes,
            channels=channels,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
        )
        == expected
    )


@pytest.mark.parametrize(
    "axes, target_axes, channels, n_channels_in, n_channels_out, error_match",
    INVALID_CASES,
)
def test_validate_channel_dim_invalid_cases(
    axes: str,
    target_axes: str | None,
    channels: list[int] | None,
    n_channels_in: int | None,
    n_channels_out: int | None,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        _validate_channel_dim(
            axes=axes,
            target_axes=target_axes,
            channels=channels,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
        )
