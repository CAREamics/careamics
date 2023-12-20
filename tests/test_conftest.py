import copy

import pytest

from careamics import Configuration


def _instantiate_without_key(config: dict, key: str):
    if isinstance(config[key], dict):
        for k in config[key]:
            _instantiate_without_key(config[key], k)
    else:
        # copy the dict
        new_config = copy.deepcopy(config)

        # remove the key
        new_config[key] = None

        # instantiate configuration
        with pytest.raises(ValueError):
            Configuration(**new_config)


def test_minimum_config(minimum_config):
    """
    Test that the minimum config is indeed a minimal example.

    First we check that we can instantiate a Configuration, then we test each
    key in the dictionary by removing it and checking that it raises an error.
    """
    # test if the configuration is valid
    Configuration(**minimum_config)

    for key in minimum_config:
        _instantiate_without_key(minimum_config, key)
