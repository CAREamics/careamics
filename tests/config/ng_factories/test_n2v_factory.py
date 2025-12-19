import pytest

from careamics.config.ng_configs import N2VConfiguration
from careamics.config.ng_factories import create_n2v_configuration
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
)


class TestN2VConfiguration:
    def test_create_configuration(self):
        """Test that N2V configuration can be created."""
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            train_dataloader_params={"num_workers": 2},
        )
        assert isinstance(config, N2VConfiguration)

    def test_no_aug(self):
        """Test the default n2v transforms."""
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            augmentations=[],
        )
        assert config.data_config.transforms == []

    def test_n2v2_structn2v(self):
        """Test that N2V2 and structn2v params are passed correctly."""
        use_n2v2 = True
        roi_size = 15
        masked_pixel_percentage = 0.5
        struct_mask_axis = SupportedStructAxis.HORIZONTAL.value
        struct_n2v_span = 15

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            use_n2v2=use_n2v2,  # median strategy
            roi_size=roi_size,
            masked_pixel_percentage=masked_pixel_percentage,
            struct_n2v_axis=struct_mask_axis,
            struct_n2v_span=struct_n2v_span,
        )
        assert (
            config.algorithm_config.n2v_config.strategy
            == SupportedPixelManipulation.MEDIAN.value
        )
        assert config.algorithm_config.n2v_config.roi_size == roi_size
        assert (
            config.algorithm_config.n2v_config.masked_pixel_percentage
            == masked_pixel_percentage
        )
        assert config.algorithm_config.n2v_config.struct_mask_axis == struct_mask_axis
        assert config.algorithm_config.n2v_config.struct_mask_span == struct_n2v_span

    def test_num_epochs(self):
        """Test that num_epochs parameter is correctly passed to trainer config."""
        num_epochs = 50

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=num_epochs,
        )
        assert (
            config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
        )

        # Test with num_epochs=None (should not be in config)
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=None,
        )
        assert "max_epochs" not in config.training_config.lightning_trainer_config

    def test_num_steps(self):
        """Test that num_steps parameter is correctly passed to trainer config."""
        num_steps = 1000

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_steps=num_steps,
        )
        assert (
            config.training_config.lightning_trainer_config["limit_train_batches"]
            == num_steps
        )

        # Test without num_steps (should not be in config)
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert (
            "limit_train_batches" not in config.training_config.lightning_trainer_config
        )

    def test_num_epochs_and_num_steps(self):
        """Test that both num_epochs and num_steps can be set simultaneously."""
        num_epochs = 25
        num_steps = 500

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=num_epochs,
            num_steps=num_steps,
        )
        assert (
            config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
        )
        assert (
            config.training_config.lightning_trainer_config["limit_train_batches"]
            == num_steps
        )

    def test_trainer_params(self):
        """Test that various trainer_params are correctly passed to trainer config."""
        trainer_params = {
            "accelerator": "gpu",
            "devices": 2,
            "precision": 16,
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 4,
            "check_val_every_n_epoch": 5,
            "log_every_n_steps": 10,
            "enable_checkpointing": True,
            "enable_progress_bar": False,
            "enable_model_summary": True,
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            trainer_params=trainer_params,
        )

        for key, value in trainer_params.items():
            assert config.training_config.lightning_trainer_config[key] == value

    def test_trainer_params_with_timing(self):
        """Test trainer_params with timing-related parameters."""
        trainer_params = {
            "min_epochs": 5,
            "min_steps": 100,
            "max_time": "00:01:00:00",  # 1 hour
            "val_check_interval": 0.25,
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            trainer_params=trainer_params,
        )

        for key, value in trainer_params.items():
            assert config.training_config.lightning_trainer_config[key] == value

    def test_trainer_params_override(self):
        """Test that explicit parameters override trainer_params."""
        num_epochs = 30
        num_steps = 800

        # trainer_params has conflicting values
        trainer_params = {
            "max_epochs": 100,
            "limit_train_batches": 0.5,
            "accelerator": "cpu",
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=num_epochs,
            num_steps=num_steps,
            trainer_params=trainer_params,
        )

        # Explicit parameters should override trainer_params
        assert (
            config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
        )
        assert (
            config.training_config.lightning_trainer_config["limit_train_batches"]
            == num_steps
        )

        # Non-conflicting trainer_params should be preserved
        assert config.training_config.lightning_trainer_config["accelerator"] == "cpu"

    def test_trainer_params_profiler(self):
        """Test trainer_params with profiler settings."""
        trainer_params = {
            "profiler": "simple",
            "detect_anomaly": True,
            "benchmark": False,
            "deterministic": True,
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            trainer_params=trainer_params,
        )

        for key, value in trainer_params.items():
            assert config.training_config.lightning_trainer_config[key] == value

    def test_trainer_params_distributed(self):
        """Test trainer_params with distributed training settings."""
        trainer_params = {
            "strategy": "ddp",
            "num_nodes": 2,
            "sync_batchnorm": True,
            "use_distributed_sampler": True,
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            trainer_params=trainer_params,
        )

        for key, value in trainer_params.items():
            assert config.training_config.lightning_trainer_config[key] == value

    def test_all_trainer_combinations(self):
        """Test comprehensive combination of all trainer parameters."""
        num_epochs = 15
        num_steps = 300

        trainer_params = {
            "accelerator": "auto",
            "devices": "auto",
            "precision": "32-true",
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 2,
            "check_val_every_n_epoch": 2,
            "val_check_interval": 1.0,
            "log_every_n_steps": 50,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "deterministic": False,
            "benchmark": True,
            "inference_mode": True,
            "use_distributed_sampler": True,
            "detect_anomaly": False,
            "barebones": False,
            "reload_dataloaders_every_n_epochs": 0,
        }

        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=num_epochs,
            num_steps=num_steps,
            trainer_params=trainer_params,
        )

        # Check explicit parameters
        assert (
            config.training_config.lightning_trainer_config["max_epochs"] == num_epochs
        )
        assert (
            config.training_config.lightning_trainer_config["limit_train_batches"]
            == num_steps
        )

        # Check trainer_params
        for key, value in trainer_params.items():
            assert config.training_config.lightning_trainer_config[key] == value

    def test_empty_trainer_params(self):
        """Test that empty trainer_params dict works correctly."""
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=None,
            trainer_params={},
        )

        # Should have empty dict (no trainer params)
        assert config.training_config.lightning_trainer_config == {}

    def test_trainer_params_none(self):
        """Test that trainer_params=None works correctly (default behavior)."""
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=None,
            trainer_params=None,
        )

        # Should have empty dict when trainer_params is None
        assert config.training_config.lightning_trainer_config == {}

    # TODO arguably this should be tested at the level of the data config only
    def test_czi_with_T_axes(self):
        """Test that SCTYX is accepted by N2V configuration for CZI data."""
        config = create_n2v_configuration(
            experiment_name="test",
            data_type="czi",
            axes="SCTYX",
            patch_size=[32, 64, 64],
            batch_size=8,
            n_channels=1,
        )
        assert config.data_config.axes == "SCTYX"
        assert len(config.data_config.patching.patch_size) == 3
        assert config.algorithm_config.model.conv_dims == 3

    @pytest.mark.parametrize(
        "axes, n_channels, channels, error",
        [
            # Valid cases
            ("YX", None, None, False),
            ("YX", 1, None, False),
            ("CYX", 1, None, False),
            ("CYX", 1, [0], False),
            ("CYX", 3, None, False),
            ("CYX", 3, [0, 2, 3], False),
            ("CYX", None, [0, 2, 3], False),
            # no channels allowed
            ("YX", None, [0], True),
            ("YX", 2, None, True),
            # mismatched channels
            ("YX", 1, [0, 2], True),
            ("YX", 3, [0, 2], True),
        ],
    )
    def test_channels(self, axes, n_channels, channels, error):
        """Test the various settings with channel parameters"""
        if error:
            with pytest.raises(ValueError):
                _ = create_n2v_configuration(
                    experiment_name="test",
                    data_type="tiff",
                    axes=axes,
                    patch_size=[64, 64],
                    batch_size=8,
                    n_channels=n_channels,
                    channels=channels,
                )
        else:
            config = create_n2v_configuration(
                experiment_name="test",
                data_type="tiff",
                axes=axes,
                patch_size=[64, 64],
                batch_size=8,
                n_channels=n_channels,
                channels=channels,
            )

            if channels is None:
                if n_channels is None:
                    n_channels = 1  # default to 1 channel

                assert config.algorithm_config.model.in_channels == n_channels
                assert config.algorithm_config.model.num_classes == n_channels
            else:
                assert config.algorithm_config.model.in_channels == len(channels)
                assert config.algorithm_config.model.num_classes == len(channels)
