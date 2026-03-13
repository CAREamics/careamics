"""Data configuration fixtures."""

from collections.abc import Callable
from functools import partial
from typing import Union

import pytest

# TODO reassess at a later point whether these fixtures should be moved to a
# sub-conftest.py, depending on their scope usage.


# TODO can probably be used to create standard configs for e2e tests
@pytest.fixture
def minimum_train_data_cfg() -> Callable:
    """Minimum dict required to create a training data configuration.

    This fixture is used to create a minimum configuration dictionary that can be
    created with any additional fields or overridden fields.

    Returns
    -------
    Callable
        A function that takes additional fields as keyword arguments and returns a
        complete training data configuration dictionary.
    """

    def _minimum_train_data_cfg(**kwargs) -> dict:
        """Create a minimum training data configuration dictionary.

        Parameters
        ----------
        **kwargs
            Additional fields to add to the minimum configuration dictionary, or fields
            to override in the minimum configuration dictionary.

        Returns
        -------
        dict
            A complete training data configuration dictionary.
        """
        config = {
            "mode": "training",
            "data_type": "array",
            "axes": "YX",
            "patching": {"name": "stratified", "patch_size": [64, 64]},
            "normalization": {"name": "mean_std"},
        }
        config.update(kwargs)
        return config

    return _minimum_train_data_cfg


# TODO probably only used for NGConfig tests, so it can be moved to a sub-conftest.py
@pytest.fixture
def minimum_mode_data_cfg(minimum_train_data_cfg) -> Callable:
    """Minimum dict required to create a data configuration for a given mode.

    This fixture ensures valid patching strategy for each mode, avoiding validation
    errors in tests passing modes along different parameters.

    Returns
    -------
    Callable
        A function that takes the mode and additional fields as keyword arguments and
        returns a complete data configuration dictionary.
    """

    def _minimum_data_mode_cfg(mode: str, **kwargs) -> dict:
        match mode:
            case "validating":
                patching = {
                    "name": "fixed_random",
                    "patch_size": [64, 64],
                }
            case "predicting":
                patching = {
                    "name": "whole",
                }
            case _:
                patching = {
                    "name": "stratified",
                    "patch_size": [64, 64],
                }

        return partial(minimum_train_data_cfg, mode=mode, patching=patching, **kwargs)()

    return _minimum_data_mode_cfg


# TODO probably only used for NGConfig tests, so it can be moved to a sub-conftest.py
@pytest.fixture
def patching_config() -> Callable:
    """Default patching configuration fixture.

    Returns
    -------
    Callable
        A function that takes the patching strategy name and whether to use 3D patching,
        and returns a complete patching configuration object.
    """
    from careamics.config.data.patching_strategies import (
        FixedRandomPatchingConfig,
        RandomPatchingConfig,
        StratifiedPatchingConfig,
        TiledPatchingConfig,
        WholePatchingConfig,
    )

    def _patching_config(name: str, is_3D: bool = False) -> Union[
        RandomPatchingConfig,
        StratifiedPatchingConfig,
        FixedRandomPatchingConfig,
        TiledPatchingConfig,
        WholePatchingConfig,
    ]:
        config = {
            "name": name,
        }

        if name in ["random", "stratified", "fixed_random", "tiled"]:
            config["patch_size"] = [64, 64, 64] if is_3D else [64, 64]

        if name == "tiled":
            config["overlaps"] = [16, 16, 16] if is_3D else [16, 16]

        match name:
            case "random":
                return RandomPatchingConfig(**config)
            case "stratified":
                return StratifiedPatchingConfig(**config)
            case "fixed_random":
                return FixedRandomPatchingConfig(**config)
            case "tiled":
                return TiledPatchingConfig(**config)
            case "whole":
                return WholePatchingConfig(**config)
            case _:
                raise ValueError(f"Invalid patching strategy name: {name}")

    return _patching_config


# TODO probably only used for NGConfig tests, so it can be moved to a sub-conftest.py
@pytest.fixture
def patch_filter_config() -> Callable:
    """Default patch filter configuration fixture.

    Returns
    -------
    Callable
        A function that takes the patch filter strategy name and returns a complete
        patch filter configuration object.
    """
    from careamics.config.data.patch_filter import (
        MaxFilterConfig,
        MeanSTDFilterConfig,
        ShannonFilterConfig,
    )

    def _patch_filter_config(
        name: str,
    ) -> Union[ShannonFilterConfig, MeanSTDFilterConfig, MaxFilterConfig]:
        config = {
            "name": name,
        }

        if name in ["shannon", "max"]:
            config["threshold"] = 0.5
        elif name == "mean_std":
            config["mean_threshold"] = 0.5
            config["std_threshold"] = 0.5

        match name:
            case "shannon":
                return ShannonFilterConfig(**config)
            case "mean_std":
                return MeanSTDFilterConfig(**config)
            case "max":
                return MaxFilterConfig(**config)
            case _:
                raise ValueError(f"Invalid patch filter strategy name: {name}")

    return _patch_filter_config


# TODO probably only used for NGConfig tests, so it can be moved to a sub-conftest.py
@pytest.fixture
def coord_filter_config() -> Callable:
    """Default coordinate filter configuration fixture.

    Returns
    -------
    Callable
        A function that returns a complete coordinate filter configuration object.
    """
    from careamics.config.data.patch_filter import MaskFilterConfig

    def _coord_filter_config() -> MaskFilterConfig:
        return MaskFilterConfig()

    return _coord_filter_config
