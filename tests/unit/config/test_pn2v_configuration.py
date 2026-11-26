"""Unit tests for PN2VConfiguration."""

import pytest

from careamics.config.algorithms import PN2VAlgorithm
from careamics.config.factories.config_discriminators import instantiate_config
from careamics.config.pn2v_configuration import PN2VConfiguration
from tests.utils import unet_config_dict_testing


def _pn2v_dict(**kwargs):
    """Return a dictionary that instantiates as PN2VConfiguration."""
    algo_kwargs = {"loss": "pn2v"}
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


def test_pn2v_mask_pixel_percentage():
    """Too few masked pixels for the patch size is rejected."""
    config = _pn2v_dict(
        patch_size=(8, 8),
        algo_kwargs={"n2v_config": {"masked_pixel_percentage": 0.05}},
    )
    with pytest.raises(ValueError, match="blind-spot"):
        instantiate_config(config)


@pytest.mark.parametrize("independent_channels", [True, False])
def test_pn2v_multichannel_config(independent_channels: bool):
    """Two input channels with four prior samples each instantiate."""
    config = _pn2v_dict(
        n_channels_in=2,
        n_channels_out=8,
        axes="CYX",
        data_kwargs={"channels": [0, 1]},
    )
    config["algorithm_config"]["model"]["independent_channels"] = independent_channels

    cfg = instantiate_config(config)

    assert isinstance(cfg, PN2VConfiguration)
    assert cfg.algorithm_config.n_prior_samples == 4


@pytest.mark.parametrize(
    "normalization",
    [
        {"name": "quantile", "per_channel": True},
        {"name": "mean_std", "input_means": [1.0], "input_stds": [2.0]},
        {"name": "min_max", "input_mins": [0.0], "input_maxes": [10.0]},
    ],
)
def test_pn2v_normalization_validated_against_input_channels(normalization):
    """Prior samples are not output channels: per-channel normalization only has to
    match the input channels."""
    config = _pn2v_dict(data_kwargs={"normalization": normalization})

    cfg = instantiate_config(config)

    assert cfg.data_config.normalization.name == normalization["name"]


@pytest.mark.parametrize(
    "metric, rejected",
    [("train_loss", False), ("train_loss_epoch", False), ("val_loss", True)],
)
def test_pn2v_monitor_with_no_validation(metric: str, rejected: bool):
    """Without validation patches, PN2V must monitor a training metric."""
    config = _pn2v_dict(
        algo_kwargs={"monitor_metric": metric},
        data_kwargs={"n_val_patches": 0},
    )

    if rejected:
        with pytest.raises(ValueError, match="No validation data"):
            instantiate_config(config)
    else:
        instantiate_config(config)


def test_pn2v_target_axes_must_be_none():
    """PN2V rejects target axes, like N2V."""
    config = _pn2v_dict(data_kwargs={"target_axes": "YX"})

    with pytest.raises(ValueError, match="PN2V does not support"):
        instantiate_config(config)
