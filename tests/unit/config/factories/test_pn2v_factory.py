"""Tests for the PN2V configuration factory."""

from careamics.config.factories import create_advanced_pn2v_config
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.pn2v_configuration import PN2VConfiguration


def test_create_pn2v_config_sets_sample_head_and_unet_params():
    """Convallaria UNet settings are config fields, not a different architecture."""
    config = create_advanced_pn2v_config(
        experiment_name="pn2v_test",
        data_type="array",
        axes="YX",
        patch_size=[64, 64],
        batch_size=1,
        noise_model=GaussianMixtureNMConfig(n_gaussian=3, n_coeff=2, min_sigma=50),
        num_classes=8,
        model_params={"depth": 3, "num_channels_init": 64},
        masked_pixel_percentage=3.125,
        roi_size=5,
        num_steps=5,
        trainer_params={"accumulate_grad_batches": 20},
    )

    assert isinstance(config, PN2VConfiguration)
    assert config.algorithm_config.model.num_classes == 8
    assert config.algorithm_config.model.in_channels == 1
    assert config.algorithm_config.model.depth == 3
    assert config.algorithm_config.model.num_channels_init == 64
    assert config.algorithm_config.n2v_config.roi_size == 5
    assert config.algorithm_config.n2v_config.masked_pixel_percentage == 3.125
    assert config.algorithm_config.noise_model.n_gaussian == 3
    assert config.algorithm_config.noise_model.min_sigma == 50
    assert config.training_config.trainer_params["limit_train_batches"] == 5
    assert config.training_config.trainer_params["accumulate_grad_batches"] == 20
    assert config.training_config.early_stopping_params is None
