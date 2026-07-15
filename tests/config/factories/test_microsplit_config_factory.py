import pytest

from careamics.config.algorithms import MicroSplitAlgorithm
from careamics.config.data import MicroSplitDataConfig
from careamics.config.factories import (
    create_advanced_microsplit_config,
    create_microsplit_config,
)
from careamics.config.microsplit_configuration import MicroSplitConfiguration

# TODO refactor similarly to https://github.com/CAREamics/careamics/pull/1005


class TestMicroSplitConfig:
    """Test the MicroSplit configuration factory."""

    def test_create_standard_config(self):
        """Test that a MicroSplit configuration can be created."""
        with pytest.warns(UserWarning):
            config = create_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
            )
        assert isinstance(config, MicroSplitConfiguration)
        assert isinstance(config.algorithm_config, MicroSplitAlgorithm)
        assert isinstance(config.data_config, MicroSplitDataConfig)

    def test_multiscale_count_propagated(self):
        """Test that multiscale_count is set on both the model and data config."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                multiscale_count=2,
            )
        assert config.algorithm_config.model.multiscale_count == 2
        assert config.data_config.multiscale_count == 2

    def test_loss_weights_propagated(self):
        """Test that the loss weights are set on the loss config."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                musplit_weight=0.3,
                denoisplit_weight=0.7,
            )
        assert config.algorithm_config.loss.musplit_weight == 0.3
        assert config.algorithm_config.loss.denoisplit_weight == 0.7

    def test_predict_logvar_consistency(self):
        """Test that model and loss agree on predict_logvar."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                predict_logvar=False,
            )
        assert config.algorithm_config.model.predict_logvar is False
        assert config.algorithm_config.loss.predict_logvar is False

    def test_supervised_checkpointing(self):
        """Test that MicroSplit uses the supervised checkpoint preset (early stop)."""
        with pytest.warns(UserWarning):
            config = create_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
            )
        assert config.training_config.early_stopping_params == {
            "monitor": "val_loss",
            "mode": "min",
        }
        assert config.training_config.checkpoint_params["monitor"] == "val_loss"

    def test_num_epochs_and_num_steps(self):
        """Test that both num_epochs and num_steps can be set simultaneously."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                num_epochs=25,
                num_steps=500,
            )
        assert config.training_config.trainer_params["max_epochs"] == 25
        assert config.training_config.trainer_params["limit_train_batches"] == 500

    def test_model_params_defaults(self):
        """Test that the MicroSplit LVAE defaults are applied to the model."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
            )
        model = config.algorithm_config.model
        assert model.z_dims == [128, 128]
        assert model.encoder_n_filters == 32
        assert model.decoder_n_filters == 32

    def test_model_params_override(self):
        """Test that model_params overrides defaults but not structural params."""
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                multiscale_count=3,
                model_params={"encoder_n_filters": 64, "encoder_dropout": 0.2},
            )
        model = config.algorithm_config.model
        # user override wins over the default
        assert model.encoder_n_filters == 64
        assert model.encoder_dropout == 0.2
        # structural params still come from the dedicated arguments
        assert model.output_channels == 2
        assert model.multiscale_count == 3
