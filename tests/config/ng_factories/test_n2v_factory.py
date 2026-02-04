import pytest

from careamics.config.ng_configs import N2VConfiguration
from careamics.config.ng_factories import (
    create_advanced_n2v_config,
    create_n2v_config,
    create_structn2v_config,
)
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
)


class TestN2VConfiguration:
    def test_create_standard_config(self):
        """Test that N2V configuration can be created."""
        config = create_n2v_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
        )
        assert isinstance(config, N2VConfiguration)

    def test_no_aug(self):
        """Test the default n2v transforms."""
        config = create_n2v_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            augmentations=[],
        )
        assert config.data_config.transforms == []

    def test_structn2v(self):
        """Test structn2v params are passed correctly."""
        struct_mask_axis = SupportedStructAxis.HORIZONTAL.value
        struct_n2v_span = 15

        config = create_structn2v_config(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            struct_n2v_axis=struct_mask_axis,
            struct_n2v_span=struct_n2v_span,
        )
        assert (
            config.algorithm_config.n2v_config.strategy
            == SupportedPixelManipulation.MEDIAN.value
        )
        assert config.algorithm_config.n2v_config.struct_mask_axis == struct_mask_axis
        assert config.algorithm_config.n2v_config.struct_mask_span == struct_n2v_span

    def test_num_epochs(self):
        """Test that num_epochs parameter is correctly passed to trainer config."""
        num_epochs = 50

        config = create_n2v_config(
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
        config = create_n2v_config(
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

        config = create_n2v_config(
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
        config = create_n2v_config(
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

        config = create_n2v_config(
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

    # TODO arguably this should be tested at the level of the data config only
    def test_czi_with_T_axes(self):
        """Test that SCTYX is accepted by N2V configuration for CZI data."""
        config = create_n2v_config(
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
                _ = create_advanced_n2v_config(
                    experiment_name="test",
                    data_type="tiff",
                    axes=axes,
                    patch_size=[64, 64],
                    batch_size=8,
                    n_channels=n_channels,
                    channels=channels,
                )
        else:
            config = create_advanced_n2v_config(
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
