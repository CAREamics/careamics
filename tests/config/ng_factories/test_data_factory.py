import pytest

from careamics.config.data import NGDataConfig
from careamics.config.data.patching_strategies import RandomPatchingConfig
from careamics.config.ng_factories.data_factory import (
    create_ng_data_configuration,
    list_spatial_augmentations,
)
from careamics.config.support import SupportedTransform
from careamics.config.transformations import (
    N2VManipulateConfig,
    XYFlipConfig,
    XYRandomRotate90Config,
)


class TestSpatialAugmentations:

    def test_list_aug_default(self):
        """Test that the default augmentations are present."""
        list_aug = list_spatial_augmentations(augmentations=None)

        assert len(list_aug) == 2
        assert list_aug[0].name == SupportedTransform.XY_FLIP.value
        assert list_aug[1].name == SupportedTransform.XY_RANDOM_ROTATE90.value

    def test_list_aug_no_aug(self):
        """Test that disabling augmentation results in empty transform list."""
        list_aug = list_spatial_augmentations(augmentations=[])
        assert len(list_aug) == 0

    def test_list_aug_error_duplicate_transforms(self):
        """Test that an error is raised when there are duplicate transforms."""
        with pytest.raises(ValueError):
            list_spatial_augmentations(
                augmentations=[
                    XYFlipConfig(),
                    XYRandomRotate90Config(),
                    XYFlipConfig(),
                ],
            )

    def test_list_aug_error_wrong_transform(self):
        """Test that an error is raised when the wrong transform is passed."""
        with pytest.raises(ValueError):
            list_spatial_augmentations(
                augmentations=[XYFlipConfig(), N2VManipulateConfig()],
            )


class TestNGDataConfiguration:

    def test_default_aug(self):
        """Test that the default augmentations are present in the configuration."""
        config: NGDataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            augmentations=None,
        )

        assert config.transforms == list_spatial_augmentations()

    def test_train_dataloader_params(self):
        """Test that shuffle is added silently to the train_dataloader_params."""
        config: NGDataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            train_dataloader_params={"num_workers": 4},
        )

        assert "shuffle" in config.train_dataloader_params
        assert config.train_dataloader_params["shuffle"]

    def test_default_patching(self):
        """Test that the default patching strategy is random."""
        config: NGDataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(32, 32),
            batch_size=2,
        )

        assert isinstance(config.patching, RandomPatchingConfig)
