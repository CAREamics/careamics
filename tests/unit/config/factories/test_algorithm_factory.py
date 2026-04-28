import pytest

from careamics.config.configuration import Configuration
from careamics.config.data.patch_filter import (
    MaxPatchFilterConfig,
    MeanStdPatchFilterConfig,
    ShannonPatchFilterConfig,
)
from careamics.config.factories import (
    create_advanced_care_config,
    create_advanced_n2n_config,
    create_advanced_n2v_config,
)
from careamics.config.factories.data_factory import SupportedPatchFilterConfig
from careamics.config.n2v_configuration import N2VConfiguration

PATCH_FILTER_CONFIGS = (
    MeanStdPatchFilterConfig(mean_threshold=0.5),
    ShannonPatchFilterConfig(threshold=0.5),
    MaxPatchFilterConfig(threshold=0.5),
)


@pytest.mark.parametrize("patch_filter_config", PATCH_FILTER_CONFIGS)
def test_advanced_care_config_patch_filter(
    patch_filter_config: SupportedPatchFilterConfig,
):
    """Test that CARE patch filter parameters are passed to the data config."""
    config: Configuration = create_advanced_care_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter_config=patch_filter_config,
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == patch_filter_config.name


@pytest.mark.parametrize("patch_filter_config", PATCH_FILTER_CONFIGS)
def test_advanced_n2n_config_patch_filter(
    patch_filter_config: SupportedPatchFilterConfig,
):
    """Test that N2N patch filter parameters are passed to the data config."""
    config: Configuration = create_advanced_n2n_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter_config=patch_filter_config,
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == "mean_std"
    assert config.data_config.patch_filter.name == patch_filter_config.name


@pytest.mark.parametrize("patch_filter_config", PATCH_FILTER_CONFIGS)
def test_advanced_n2v_config_patch_filter(
    patch_filter_config: SupportedPatchFilterConfig,
):
    """Test that N2V patch filter parameters are passed to the data config."""
    config: N2VConfiguration = create_advanced_n2v_config(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        patch_filter_config=patch_filter_config,
    )

    assert config.data_config.patch_filter is not None
    assert config.data_config.patch_filter.name == patch_filter_config.name
