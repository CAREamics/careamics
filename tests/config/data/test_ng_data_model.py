import pytest

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)


@pytest.mark.parametrize(
    "patching_strategy",
    [
        {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
        {
            "name": SupportedPatchingStrategy.TILED,
            "patch_size": [16, 16],
            "overlaps": [4, 4],
        },
        {"name": SupportedPatchingStrategy.WHOLE},
    ],
)
def test_ng_data_config_strategy(patching_strategy):
    data_config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching=patching_strategy,
        normalization={"name": "standardize"},
    )
    assert data_config.patching.name == patching_strategy["name"]


@pytest.mark.parametrize(
    "axes, patching_strategy",
    [
        ("ZYX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]}),
        ("YX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16, 16]}),
        (
            "ZYX",
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
        ),
        (
            "SYX",
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16, 16],
                "overlaps": [4, 4, 4],
            },
        ),
    ],
)
def test_ng_dataset_invalid_axes_patch(axes, patching_strategy):
    with pytest.raises(ValueError):
        NGDataConfig(
            data_type="array",
            axes=axes,
            patching=patching_strategy,
            normalization={"name": "standardize"},
        )


@pytest.mark.parametrize(
    "normalization",
    [
        {"name": "standardize"},
        {"name": "standardize", "input_means": [0.5], "input_stds": [0.2]},
        {"name": "none"},
        {"name": "minmax"},
        {"name": "minmax", "input_mins": [0.0], "input_maxes": [255.0]},
        {"name": "quantile"},
        {"name": "quantile", "lower_quantile": 0.01, "upper_quantile": 0.99},
    ],
)
def test_ng_data_config_normalization(normalization):
    data_config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching={"name": "whole"},
        normalization=normalization,
    )
    assert data_config.normalization.name == normalization["name"]


@pytest.mark.parametrize(
    "normalization",
    [
        {"name": "standardize", "input_means": [0.5]},  # missing stds
        {"name": "minmax", "input_mins": [0.0]},  # missing maxes
        {
            "name": "quantile",
            "lower_quantile": 0.99,
            "upper_quantile": 0.01,
        },  # wrong order
    ],
)
def test_ng_data_config_invalid_normalization(normalization):
    with pytest.raises(ValueError):
        NGDataConfig(
            data_type="array",
            axes="YX",
            patching={"name": "whole"},
            normalization=normalization,
        )
