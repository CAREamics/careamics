"""Tests for CryoCARE data configuration."""

import pytest
from pydantic import ValidationError

from careamics.config import CryoCAREDataConfig
from careamics.config.augmentations import (
    TiltAxisRandomRotate90Config,
)
from careamics.config.data.normalization_config import NoNormConfig


def _cryocare_data(**kwargs) -> dict:
    data = {
        "mode": "training",
        "data_type": "array",
        "axes": "ZYX",
        "tilt_axis": "Z",
        "patching": {"name": "stratified", "patch_size": (32, 32, 32)},
        "normalization": NoNormConfig(),
        "batch_size": 1,
    }
    data.update(kwargs)
    return data


@pytest.mark.parametrize(
    "overrides",
    [
        {"axes": "YX"},
        {"axes": "CZYX"},
        {"channels": [0]},
        {"patching": {"name": "random", "patch_size": (32, 32, 32)}},
        {"patching": {"name": "stratified", "patch_size": (32, 32)}},
    ],
)
def test_cryocare_data_constraints(overrides: dict) -> None:
    """Reject data configurations incompatible with CryoCARE."""
    with pytest.raises(ValidationError):
        CryoCAREDataConfig(**_cryocare_data(**overrides))


def test_default_augmentations_include_rotation_for_compatible_patch() -> None:
    """Add rotation and swapping when the patch supports the tilt-axis rotation."""
    config = CryoCAREDataConfig(**_cryocare_data())

    assert [augmentation.name for augmentation in config.augmentations] == [
        "TiltAxisRandomRotate90",
        "SwapInputTarget",
    ]
    assert config.augmentations[0].tilt_axis == "Z"


def test_default_augmentations_only_swap_for_incompatible_patch() -> None:
    """Only add swapping when the patch cannot support the tilt-axis rotation."""
    config = CryoCAREDataConfig(
        **_cryocare_data(
            tilt_axis="Y",
            patching={"name": "stratified", "patch_size": (16, 32, 32)},
        )
    )

    assert [augmentation.name for augmentation in config.augmentations] == [
        "SwapInputTarget"
    ]


def test_explicit_empty_augmentations_are_preserved() -> None:
    """Do not add defaults when augmentations are explicitly disabled."""
    config = CryoCAREDataConfig(**_cryocare_data(augmentations=[]))

    assert config.augmentations == []


def test_rejects_rotation_with_different_tilt_axis() -> None:
    """Reject rotations whose tilt axis differs from the data configuration."""
    with pytest.raises(ValidationError, match="different"):
        CryoCAREDataConfig(
            **_cryocare_data(
                augmentations=[TiltAxisRandomRotate90Config(tilt_axis="Y")]
            )
        )


def test_rejects_rotation_with_incompatible_patch() -> None:
    """Reject rotations when perpendicular patch dimensions are unequal."""
    with pytest.raises(ValidationError, match="equal patch dimensions"):
        CryoCAREDataConfig(
            **_cryocare_data(
                tilt_axis="Y",
                patching={"name": "stratified", "patch_size": (16, 32, 32)},
                augmentations=[TiltAxisRandomRotate90Config(tilt_axis="Y")],
            )
        )
