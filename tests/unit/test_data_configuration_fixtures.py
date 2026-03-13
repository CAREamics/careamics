"""Tests for the data configuration fixtures.

These fixtures can be found in `tests/fixtures/data_configurations.py`.
"""

import pytest


def test_minimum_data_cfg(minimum_train_data_cfg):
    """Test the minimum data configuration fixture."""
    from careamics.config.data.ng_data_config import NGDataConfig

    cfg_dict = minimum_train_data_cfg()
    NGDataConfig(**cfg_dict)


@pytest.mark.parametrize("mode", ["training", "validating", "predicting"])
@pytest.mark.parametrize("batch_size", [3])
def test_minimum_data_mode_cfg(minimum_mode_data_cfg, mode, batch_size):
    """Test the minimum data configuration fixture for each mode."""
    from careamics.config.data.ng_data_config import NGDataConfig

    cfg_dict = minimum_mode_data_cfg(mode=mode, batch_size=batch_size)
    cfg = NGDataConfig(**cfg_dict)
    assert cfg.mode == mode
    assert cfg.batch_size == batch_size


@pytest.mark.parametrize(
    "name", ["random", "stratified", "fixed_random", "tiled", "whole"]
)
def test_patching_config(patching_config, name):
    """Test the patching configuration fixture."""
    cfg = patching_config(name)
    assert cfg.name == name


@pytest.mark.parametrize("name", ["shannon", "mean_std", "max"])
def test_patch_filter_config(patch_filter_config, name):
    """Test the patch filter configuration fixture."""
    from careamics.config.data.patch_filter import FilterConfig

    cfg = patch_filter_config(name)
    assert isinstance(cfg, FilterConfig)
    assert cfg.name == name


def test_coord_filter_config(coord_filter_config):
    """Test the coordinate filter configuration fixture."""
    from careamics.config.data.patch_filter import FilterConfig

    cfg = coord_filter_config()
    assert isinstance(cfg, FilterConfig)
    assert cfg.name == "mask"
