import pytest

from careamics.config.ng_configs import CAREConfiguration
from careamics.config.ng_factories import (
    create_advanced_care_config,
    create_care_config,
)


class TestStandardConfig:
    """Test the standard CARE configuration factory."""

    def test_create_standard_config(self):
        """Test that CARE configuration can be created."""
        config = create_care_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert isinstance(config, CAREConfiguration)

    def test_no_aug(self):
        """Test no augmentation."""
        config = create_care_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            augmentations=[],
        )
        assert config.data_config.transforms == []

    def test_num_epochs_and_num_steps(self):
        """Test that both num_epochs and num_steps can be set simultaneously."""
        num_epochs = 25
        num_steps = 500

        config = create_care_config(
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


class TestAdvancedCAREConfig:
    """Test the advanced CARE configuration factory"""

    @pytest.mark.parametrize(
        "axes, n_channels_in, n_channels_out, channels, error",
        [
            # Valid cases
            ("YX", None, None, None, False),
            ("YX", 1, None, None, False),
            ("YX", 1, 1, None, False),
            ("CYX", 1, None, None, False),
            ("CYX", 1, 1, None, False),
            ("CYX", 1, None, [0], False),
            ("CYX", 1, 1, [0], False),
            ("CYX", 3, None, None, False),
            ("CYX", 3, 3, [0, 2, 3], False),
            ("CYX", 3, 2, [0, 2, 3], False),
            # no channels allowed
            ("YX", None, None, [0], True),
            ("YX", 2, None, None, True),
            ("YX", 2, 1, None, True),
            ("YX", 1, 2, None, True),
            ("YX", None, 2, None, True),
            # mismatched channels
            ("YX", 1, None, [0, 2], True),
            ("YX", 1, 2, [0, 2], True),
            ("YX", 3, None, [0, 2], True),
            ("YX", 3, 1, [0, 2], True),
        ],
    )
    def test_channels(self, axes, n_channels_in, n_channels_out, channels, error):
        """Test the various settings with channel parameters"""
        if error:
            with pytest.raises(ValueError):
                _ = create_advanced_care_config(
                    experiment_name="test",
                    data_type="tiff",
                    axes=axes,
                    patch_size=[64, 64],
                    batch_size=8,
                    n_channels_in=n_channels_in,
                    n_channels_out=n_channels_out,
                    channels=channels,
                )
        else:
            config = create_advanced_care_config(
                experiment_name="test",
                data_type="tiff",
                axes=axes,
                patch_size=[64, 64],
                batch_size=8,
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                channels=channels,
            )

            exp_channel_in = n_channels_in
            exp_channel_out = n_channels_out

            if channels is None:
                if exp_channel_in is None:
                    exp_channel_in = 1  # default to 1 channel
                if exp_channel_out is None:
                    exp_channel_out = exp_channel_in  # default to exp_channel_in

                assert config.algorithm_config.model.in_channels == exp_channel_in
                assert config.algorithm_config.model.num_classes == exp_channel_out
            else:
                if exp_channel_in is None:
                    exp_channel_in = len(channels)
                if exp_channel_out is None:
                    exp_channel_out = len(channels)
                assert config.algorithm_config.model.in_channels == exp_channel_in
                assert config.algorithm_config.model.num_classes == exp_channel_out

    def test_num_workers(self):
        """Test that num_workers can be set and overrriden by train dataloader."""
        num_workers = 4

        config: CAREConfiguration = create_advanced_care_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_workers=num_workers,
        )
        assert config.data_config.train_dataloader_params["num_workers"] == num_workers
        assert config.data_config.val_dataloader_params["num_workers"] == num_workers
        assert config.data_config.pred_dataloader_params["num_workers"] == num_workers

        # test overrride
        alt_num_workers = 2
        config: CAREConfiguration = create_advanced_care_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_workers=num_workers,
            train_dataloader_params={"num_workers": alt_num_workers},
        )
        assert (
            config.data_config.train_dataloader_params["num_workers"] == alt_num_workers
        )
        assert config.data_config.val_dataloader_params["num_workers"] == num_workers
        assert config.data_config.pred_dataloader_params["num_workers"] == num_workers
