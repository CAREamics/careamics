"""Tests for the PN2V configuration factory."""

import pytest

from careamics.config.factories import create_advanced_pn2v_config, create_pn2v_config
from careamics.config.pn2v_configuration import PN2VConfiguration


def test_create_pn2v_config_sets_sample_head_and_unet_params():
    """Convallaria UNet settings are config fields, not a different architecture."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=1,
        n_prior_samples=8,
        model_params={"depth": 3, "num_channels_init": 64},
        masked_pixel_percentage=3.125,
        roi_size=5,
        num_steps=5,
        trainer_params={"accumulate_grad_batches": 20},
    )

    assert isinstance(config, PN2VConfiguration)
    assert config.algorithm_config.model.num_classes == 8
    assert config.algorithm_config.n_prior_samples == 8
    assert config.algorithm_config.model.in_channels == 1
    assert config.algorithm_config.model.depth == 3
    assert config.algorithm_config.model.num_channels_init == 64
    assert config.algorithm_config.n2v_config.roi_size == 5
    assert config.algorithm_config.n2v_config.masked_pixel_percentage == 3.125
    assert config.training_config.trainer_params["limit_train_batches"] == 5
    assert config.training_config.trainer_params["accumulate_grad_batches"] == 20
    assert config.training_config.early_stopping_params is None


def test_create_pn2v_config_defaults():
    """800 prior samples and no batch normalization, as in the PN2V paper."""
    config = create_pn2v_config(
        experiment_name="pn2v_test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
    )

    assert config.algorithm_config.n_prior_samples == 800
    assert config.algorithm_config.model.num_classes == 800
    assert config.algorithm_config.model.use_batch_norm is False


@pytest.mark.parametrize("independent_channels", [True, False])
def test_create_pn2v_config_multichannel(independent_channels: bool):
    """The UNet predicts `n_prior_samples` outputs per input channel."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_test",
        data_type="array",
        axes="SCYX",
        patch_size=[64, 64],
        batch_size=8,
        n_channels=3,
        n_prior_samples=10,
        independent_channels=independent_channels,
    )

    assert config.algorithm_config.model.in_channels == 3
    assert config.algorithm_config.model.num_classes == 30
    assert config.algorithm_config.model.independent_channels is independent_channels
    assert config.algorithm_config.n_prior_samples == 10


def test_create_pn2v_config_model_params():
    """Batch normalization can be switched on and the caller's dict is not changed."""
    model_params = {"depth": 2, "use_batch_norm": True}

    config = create_advanced_pn2v_config(
        experiment_name="pn2v_test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        model_params=model_params,
        project_name="pn2v_project",
    )

    assert config.algorithm_config.model.use_batch_norm is True
    assert config.algorithm_config.model.depth == 2
    assert model_params == {"depth": 2, "use_batch_norm": True}
    assert config.project_name == "pn2v_project"
