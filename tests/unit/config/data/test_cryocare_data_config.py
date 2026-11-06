"""Tests for CryoCARE data configuration."""

import pytest
from pydantic import ValidationError

from careamics.config import CryoCAREDataConfig
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


def test_channels_must_be_none() -> None:
    """Reject channel selection for CryoCARE."""
    with pytest.raises(ValidationError, match="does not support channel selection"):
        CryoCAREDataConfig(**_cryocare_data(channels=[0]))


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


def test_augmentations_are_derived_from_configuration() -> None:
    """Ignore supplied augmentations and derive them from CryoCARE parameters."""
    config = CryoCAREDataConfig(**_cryocare_data(augmentations=[]))

    assert [augmentation.name for augmentation in config.augmentations] == [
        "TiltAxisRandomRotate90",
        "SwapInputTarget",
    ]


def test_non_training_config_uses_no_augmentations() -> None:
    """Disable CryoCARE training augmentations outside training mode."""
    config = CryoCAREDataConfig(
        **_cryocare_data(
            mode="predicting",
            patching={"name": "whole"},
        )
    )

    assert config.augmentations == []


def test_training_requires_stratified_patching() -> None:
    """Reject non-stratified patching for CryoCARE training and validation."""
    for mode, patching in (
        ("training", {"name": "random", "patch_size": (32, 32, 32)}),
        ("validating", {"name": "fixed_random", "patch_size": (32, 32, 32)}),
    ):
        with pytest.raises(ValidationError, match="require stratified patching"):
            CryoCAREDataConfig(
                **_cryocare_data(mode=mode, patching=patching),
            )
