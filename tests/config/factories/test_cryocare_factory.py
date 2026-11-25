"""Tests for the CryoCARE configuration factory."""

import pytest

from careamics.config.algorithms import CryoCAREAlgorithm
from careamics.config.cryocare_configuration import CryoCAREConfiguration
from careamics.config.data import CryoCAREDataConfig
from careamics.config.factories import (
    create_advanced_cryocare_config,
    create_cryocare_config,
)


def _create_config(**kwargs):
    parameters = {
        "experiment_name": "test",
        "data_type": "array",
        "axes": "ZYX",
        "patch_size": (32, 32, 32),
        "batch_size": 8,
        "tilt_axis": "Y",
    }
    parameters.update(kwargs)
    return create_cryocare_config(**parameters)


def test_create_standard_config() -> None:
    """Create a CryoCARE configuration with the standard factory."""
    config = _create_config()

    assert isinstance(config, CryoCAREConfiguration)
    assert isinstance(config.algorithm_config, CryoCAREAlgorithm)
    assert isinstance(config.data_config, CryoCAREDataConfig)


def test_cryo_care_specific_parameters_are_propagated() -> None:
    """Propagate the tilt-axis split parameters to the data configuration."""
    config = _create_config(validation_fraction=0.2, n_val_patches=120)

    assert config.data_config.tilt_axis == "Y"
    assert config.data_config.validation_fraction == 0.2
    assert config.data_config.n_val_patches == 120


def test_augmentations_and_channels_are_not_factory_parameters() -> None:
    """Generate augmentations and disable channel selection automatically."""
    config = create_advanced_cryocare_config(
        experiment_name="test",
        data_type="array",
        axes="ZYX",
        patch_size=(32, 32, 32),
        batch_size=8,
        tilt_axis="Y",
    )

    assert config.data_config.channels is None
    assert [augmentation.name for augmentation in config.data_config.augmentations] == [
        "TiltAxisRandomRotate90",
        "SwapInputTarget",
    ]


@pytest.mark.parametrize("n_val_patches", [0, 19])
def test_validation_patches_must_be_at_least_twenty(n_val_patches: int) -> None:
    """Reject fewer than twenty validation patches."""
    with pytest.raises(ValueError, match="greater than or equal to 20"):
        _create_config(n_val_patches=n_val_patches)
