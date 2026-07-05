import pytest

from careamics.config.algorithms import HDNAlgorithm
from careamics.config.factories import (
    create_advanced_hdn_config,
    create_hdn_config,
)
from careamics.config.hdn_configuration import HDNConfiguration


class TestHDNConfig:
    """Test the HDN configuration factory."""

    def test_create_standard_config(self):
        """Test that an HDN configuration can be created."""
        config = create_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert isinstance(config, HDNConfiguration)
        assert isinstance(config.algorithm_config, HDNAlgorithm)
        assert config.algorithm_config.model.architecture == "LVAE"

    def test_predict_logvar_from_noise_model(self):
        """Test that predict_logvar is enabled only without a noise model."""
        config = create_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert config.algorithm_config.model.predict_logvar is True
        assert config.algorithm_config.loss.predict_logvar is True

    def test_default_optimizer(self):
        """Test the default HDN optimizer."""
        config = create_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert config.algorithm_config.optimizer.name == "Adamax"
        assert config.algorithm_config.optimizer.parameters == {"lr": 3e-4}

    def test_output_channels_must_be_one(self):
        """Test that HDN rejects more than one output channel."""
        with pytest.raises(ValueError):
            create_advanced_hdn_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
            )

    def test_no_aug(self):
        """Test no augmentation."""
        config = create_advanced_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            augmentations=[],
        )
        assert config.data_config.augmentations == []

    def test_num_epochs_and_num_steps(self):
        """Test that both num_epochs and num_steps can be set simultaneously."""
        config = create_advanced_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=25,
            num_steps=500,
        )
        assert config.training_config.trainer_params["max_epochs"] == 25
        assert config.training_config.trainer_params["limit_train_batches"] == 500

    def test_self_supervised_checkpointing(self):
        """Test that HDN uses the self-supervised checkpoint preset (no early stop)."""
        config = create_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert config.training_config.early_stopping_params is None
        assert "every_n_epochs" in config.training_config.checkpoint_params

    def test_3d(self):
        """Test that a 3D HDN configuration can be created."""
        config = create_advanced_hdn_config(
            experiment_name="test",
            data_type="array",
            axes="ZYX",
            patch_size=[16, 64, 64],
            batch_size=2,
        )
        assert config.algorithm_config.model.is_3D()
        assert config.data_config.is_3D()
