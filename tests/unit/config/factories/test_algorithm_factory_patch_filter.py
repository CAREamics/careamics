from careamics.config.configuration import Configuration
from careamics.config.factories import (
    create_advanced_care_config,
    create_advanced_n2n_config,
    create_advanced_n2v_config,
)
from careamics.config.n2v_configuration import N2VConfiguration


def test_advanced_care_config_patch_filter():
    """Test that CARE patch filter parameters are passed to the data config."""
    config: Configuration = create_advanced_care_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter="max",
        patch_filter_params={"threshold": 0.5},
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == "max"
    assert config.data_config.patch_filter.threshold == 0.5


def test_advanced_n2n_config_patch_filter():
    """Test that N2N patch filter parameters are passed to the data config."""
    config: Configuration = create_advanced_n2n_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter="mean_std",
        patch_filter_params={"mean_threshold": 0.5},
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == "mean_std"
    assert config.data_config.patch_filter.mean_threshold == 0.5


def test_advanced_n2v_config_patch_filter():
    """Test that N2V patch filter parameters are passed to the data config."""
    config: N2VConfiguration = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter="shannon",
        patch_filter_params={"threshold": 0.5},
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == "shannon"
    assert config.data_config.patch_filter.threshold == 0.5
