from contextlib import nullcontext as does_not_raise

import pytest

from careamics.config.factories.factory_utils import (
    assemble_augmentations,
    validate_input_channels,
)
from careamics.config.support import SupportedAugmentation

# --- List of parameters re-used in tests


# --- List of test parameters

VALID_CASES = [
    # axes, channels, in_channels, error_match
    # no channels
    ("YX", None, None, does_not_raise()),
    ("YX", None, 1, does_not_raise()),
    ("YX", None, 1, does_not_raise()),
    # with channels
    ("CYX", [0], None, does_not_raise()),
    ("CYX", [0], 1, does_not_raise()),
    ("CYX", [0, 2], None, does_not_raise()),
    ("CYX", [0, 2], 2, does_not_raise()),
    ("CYX", None, 3, does_not_raise()),
    ("CYX", None, 1, does_not_raise()),
]

INVALID_CASES = [
    # axes, channels, in_channels, error_match
    (
        "CYX",
        None,
        None,
        pytest.raises(ValueError, match="`n_channels` or `channels` must be specified"),
    ),
    (
        "YX",
        [0],
        None,
        pytest.raises(
            ValueError,
            match=("`channels` can only be specified when `axes` includes 'C'"),
        ),
    ),
    (
        "CYX",
        [],
        None,
        pytest.raises(ValueError, match="`channels` cannot not be empty"),
    ),
    (
        "YX",
        None,
        2,
        pytest.raises(ValueError, match="C is not present in the axes"),
    ),
    (
        "CYX",
        [0, 2],
        3,
        pytest.raises(ValueError, match="does not match length of `channels`"),
    ),
]

AUGS = [
    [],
    ["x_flip"],
    ["y_flip"],
    ["rotate_90"],
    ["x_flip", "y_flip"],
    ["x_flip", "rotate_90"],
    ["y_flip", "rotate_90"],
    ["x_flip", "rotate_90", "y_flip"],
]

SEED = [None, 42]


# --- Unit tests


@pytest.mark.parametrize(
    "axes, channels, n_channels, exp_error",
    VALID_CASES + INVALID_CASES,
)
def test_validate_channel_dim_invalid_cases(
    axes: str,
    channels: list[int] | None,
    n_channels: int | None,
    exp_error,
) -> None:
    with exp_error:
        validate_input_channels(
            axes=axes,
            channels=channels,
            n_channels=n_channels,
        )


@pytest.mark.parametrize("augs", AUGS)
@pytest.mark.parametrize("seed", SEED)
def test_assemble_augmentations(augs, seed):
    """Test the assemble_augmentations function."""
    xy_flip = "x_flip" in augs or "y_flip" in augs
    rot_90 = "rotate_90" in augs

    augmentations = assemble_augmentations(augmentations=augs, seed=seed)

    is_xy_flip = any(
        aug.name == SupportedAugmentation.XY_FLIP.value for aug in augmentations
    )
    is_rot_90 = any(
        aug.name == SupportedAugmentation.XY_RANDOM_ROTATE90.value
        for aug in augmentations
    )

    for aug in augmentations:
        assert aug.seed == seed

    assert is_xy_flip == xy_flip
    assert is_rot_90 == rot_90
