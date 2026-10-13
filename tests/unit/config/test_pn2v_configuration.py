"""Unit tests for PN2VConfiguration."""

import pytest

from careamics.config.algorithms import PN2VAlgorithm
from careamics.config.factories.config_discriminators import instantiate_config
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.pn2v_configuration import PN2VConfiguration
from tests.utils import unet_config_dict_testing


def _pn2v_dict(**kwargs):
    """Return a dictionary that instantiates as PN2VConfiguration."""
    algo_kwargs = {
        "loss": "pn2v",
        "noise_model": GaussianMixtureNMConfig().model_dump(),
    }
    algo_kwargs.update(kwargs.pop("algo_kwargs", {}))
    return unet_config_dict_testing(
        algorithm="pn2v",
        n_channels_out=kwargs.pop("n_channels_out", 4),
        model_kwargs={"independent_channels": True},
        algo_kwargs=algo_kwargs,
        **kwargs,
    )


def test_pn2v_config_instantiation():
    """PN2V dictionaries instantiate as PN2VConfiguration."""
    cfg = instantiate_config(_pn2v_dict())
    assert isinstance(cfg, PN2VConfiguration)
    assert isinstance(cfg.algorithm_config, PN2VAlgorithm)
    assert cfg.algorithm_config.model.in_channels == 1
    assert cfg.algorithm_config.model.num_classes == 4
    assert cfg.algorithm_config.sample_scale == 10.0
    assert cfg.is_supervised() is False


def test_pn2v_requires_noise_model():
    """A PN2V configuration without a noise model is rejected."""
    config = _pn2v_dict()
    del config["algorithm_config"]["noise_model"]
    with pytest.raises(ValueError):
        instantiate_config(config)


def test_pn2v_mask_pixel_percentage():
    """Too few masked pixels for the patch size is rejected."""
    config = _pn2v_dict(
        patch_size=(8, 8),
        algo_kwargs={"n2v_config": {"masked_pixel_percentage": 0.05}},
    )
    with pytest.raises(ValueError, match="blind-spot"):
        instantiate_config(config)
