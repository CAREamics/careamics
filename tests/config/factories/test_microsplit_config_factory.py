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
                gaussian_likelihood_weight=0.3,
                noise_model_likelihood_weight=0.7,
            )
        assert config.algorithm_config.loss.gaussian_likelihood_weight == 0.3
        assert config.algorithm_config.loss.noise_model_likelihood_weight == 0.7

    def test_predict_logvar_consistency(self):
        """Test that model and loss agree on predict_logvar.

        `predict_logvar=False` is only valid without the muSplit Gaussian likelihood
        (`gaussian_likelihood_weight=0`), i.e. for pure denoiSplit.
        """
        with pytest.warns(UserWarning):
            config = create_advanced_microsplit_config(
                experiment_name="test",
                data_type="array",
                axes="YX",
                patch_size=[64, 64],
                batch_size=8,
                output_channels=2,
                predict_logvar=False,
                gaussian_likelihood_weight=0.0,
                noise_model_likelihood_weight=1.0,
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
        assert model.n_filters == 32

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
                model_params={"n_filters": 64, "encoder_dropout": 0.2},
            )
        model = config.algorithm_config.model
        # user override wins over the default
        assert model.n_filters == 64
        assert model.encoder_dropout == 0.2
        # structural params still come from the dedicated arguments
        assert model.output_channels == 2
        assert model.multiscale_count == 3


def _base_config() -> MicroSplitConfiguration:
    """Build a valid MicroSplit configuration for cross-validation tests."""
    with pytest.warns(UserWarning):
        return create_advanced_microsplit_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            output_channels=2,
            multiscale_count=1,
        )


def test_normalization_none_rejected():
    """Test that MicroSplit rejects disabled normalization (issue #1015)."""
    with pytest.raises(ValueError, match="requires normalized inputs"):
        create_advanced_microsplit_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            output_channels=2,
            multiscale_count=1,
            gaussian_likelihood_weight=1.0,
            noise_model_likelihood_weight=0.0,
            normalization="none",
        )


def test_alpha_ranges_must_match_output_channels():
    """Test that one alpha range per output channel is required."""
    with pytest.raises(ValueError, match="must match the number of"):
        create_advanced_microsplit_config(
            experiment_name="test",
            data_type="array",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            output_channels=2,
            multiscale_count=1,
            gaussian_likelihood_weight=1.0,
            noise_model_likelihood_weight=0.0,
            alpha_ranges=[(0.0, 1.0)],
        )


def test_multiscale_mismatch_rejected():
    """Test that the model and data multiscale_count must agree."""
    base = _base_config()
    mismatched_data = base.data_config.model_copy(update={"multiscale_count": 3})
    with pytest.raises(ValueError, match="multiscale_count"):
        MicroSplitConfiguration(
            experiment_name="test",
            algorithm_config=base.algorithm_config,
            data_config=mismatched_data,
            training_config=base.training_config,
        )


def test_input_shape_must_match_patch_size():
    """Test that the model input_shape must equal the data patch size."""
    base = _base_config()
    mismatched_model = base.algorithm_config.model.model_copy(
        update={"input_shape": (128, 128)}
    )
    algo = base.algorithm_config.model_copy(update={"model": mismatched_model})
    with pytest.raises(ValueError, match="must match the data"):
        MicroSplitConfiguration(
            experiment_name="test",
            algorithm_config=algo,
            data_config=base.data_config,
            training_config=base.training_config,
        )
