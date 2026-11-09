"""Tests for CryoCARE data configuration."""

import pytest
from pydantic import ValidationError

from careamics.config import CryoCAREDataConfig
from careamics.config.data.normalization_config import NoNormConfig
from careamics.config.data.patching_strategies import (
    FixedRandomPatchingConfig,
    WholePatchingConfig,
)


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


def test_target_axes_must_match_input_axes() -> None:
    """Reject target axes that differ from input axes."""
    with pytest.raises(ValidationError, match="target axes to match input axes"):
        CryoCAREDataConfig(**_cryocare_data(target_axes="YX"))


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
    """Reject non-stratified patching for CryoCARE training."""
    with pytest.raises(ValidationError, match="requires stratified patching"):
        CryoCAREDataConfig(
            **_cryocare_data(
                patching={"name": "random", "patch_size": (32, 32, 32)},
            ),
        )


def test_validation_accepts_fixed_random_patching() -> None:
    """Accept fixed random patching for CryoCARE validation."""
    config = CryoCAREDataConfig(
        **_cryocare_data(
            mode="validating",
            patching={"name": "fixed_random", "patch_size": (32, 32, 32)},
        ),
    )

    assert isinstance(config.patching, FixedRandomPatchingConfig)
    assert config.augmentations == []


@pytest.mark.parametrize("mode", ["validating", "predicting"])
def test_convert_mode_preserves_cryocare_fields(mode: str) -> None:
    """Preserve CryoCARE fields when converting the training config."""
    config = CryoCAREDataConfig(**_cryocare_data())

    converted = config.convert_mode(mode)  # type: ignore[arg-type]

    assert isinstance(converted, CryoCAREDataConfig)
    assert converted.tilt_axis == config.tilt_axis
    assert converted.validation_fraction == config.validation_fraction
    assert converted.augmentations == []
    if mode == "validating":
        assert isinstance(converted.patching, FixedRandomPatchingConfig)
    else:
        assert isinstance(converted.patching, WholePatchingConfig)


@pytest.mark.parametrize(
    "conversion_kwargs",
    [
        {"new_axes": "ZYX"},
        {"new_target_axes": "ZYX"},
        {"new_channels": "all"},
    ],
)
def test_convert_mode_rejects_unsupported_changes(conversion_kwargs: dict) -> None:
    """Reject changes to axes and channels during CryoCARE conversion."""
    config = CryoCAREDataConfig(**_cryocare_data())

    with pytest.raises(ValueError, match="does not support changing"):
        config.convert_mode("predicting", **conversion_kwargs)
