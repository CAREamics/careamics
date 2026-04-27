from typing import Any, Literal

import pytest

from careamics.config.algorithms.n2v_manipulation import N2VManipulateConfig
from careamics.config.augmentations import (
    XYFlipConfig,
    XYRandomRotate90Config,
)
from careamics.config.data import DataConfig
from careamics.config.data.patching_strategies import StratifiedPatchingConfig
from careamics.config.factories.data_factory import (
    create_ng_data_configuration,
    create_patch_filter_configuration,
    list_spatial_augmentations,
)
from careamics.config.support import SupportedTransform

PatchFilterName = Literal["mean_std", "shannon", "max"]
PATCH_FILTER_PARAMS: tuple[tuple[PatchFilterName, dict[str, Any]], ...] = (
    ("mean_std", {"mean_threshold": 0.5}),
    ("shannon", {"threshold": 0.5}),
    ("max", {"threshold": 0.5}),
)
PATCH_FILTER_MISSING_REQUIRED_PARAMS: tuple[
    tuple[PatchFilterName, dict[str, Any]], ...
] = (
    ("mean_std", {}),
    ("shannon", {}),
    ("max", {}),
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
                    XYFlipConfig(seed=42),
                    XYRandomRotate90Config(seed=42),
                    XYFlipConfig(seed=42),
                ],
            )

    def test_list_aug_error_wrong_transform(self):
        """Test that an error is raised when the wrong transform is passed."""
        with pytest.raises(ValueError):
            list_spatial_augmentations(
                augmentations=[XYFlipConfig(seed=42), N2VManipulateConfig(seed=42)],
            )


class TestDataConfiguration:

    def test_default_aug(self):
        """Test that the default augmentations are present in the configuration."""
        config: DataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            augmentations=None,
            seed=42,
        )

        assert config.augmentations == list_spatial_augmentations(seed=42)

    def test_train_dataloader_params(self):
        """Test that shuffle is added silently to the train_dataloader_params."""
        config: DataConfig = create_ng_data_configuration(
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
        config: DataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(32, 32),
            batch_size=2,
        )

        assert isinstance(config.patching, StratifiedPatchingConfig)

    def test_num_workers_explicit(self):
        """Test that an explicit num_workers value is passed through unchanged."""
        config: DataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            num_workers=4,
        )

        assert config.train_dataloader_params["num_workers"] == 4
        assert config.val_dataloader_params["num_workers"] == 4
        assert config.pred_dataloader_params["num_workers"] == 4

    @pytest.mark.parametrize("patch_filter, patch_filter_params", PATCH_FILTER_PARAMS)
    def test_patch_filter(
        self,
        patch_filter: PatchFilterName,
        patch_filter_params: dict[str, Any],
    ):
        """Test that patch filter configuration is passed through."""
        config: DataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            patch_filter={"name": patch_filter, **patch_filter_params},
        )

        assert config.patch_filter is not None
        assert config.patch_filter.name == patch_filter

    @pytest.mark.parametrize("patch_filter, patch_filter_params", PATCH_FILTER_PARAMS)
    def test_create_patch_filter_configuration(
        self,
        patch_filter: PatchFilterName,
        patch_filter_params: dict[str, Any],
    ):
        """Test that patch filter factory arguments create a DataConfig dictionary."""
        patch_filter_config = create_patch_filter_configuration(
            patch_filter=patch_filter,
            patch_filter_params=patch_filter_params,
        )

        assert patch_filter_config == {"name": patch_filter, **patch_filter_params}

    @pytest.mark.parametrize(
        "patch_filter, patch_filter_params",
        PATCH_FILTER_MISSING_REQUIRED_PARAMS,
    )
    def test_error_missing_patch_filter_params(
        self,
        patch_filter: PatchFilterName,
        patch_filter_params: dict[str, Any],
    ):
        """Test that missing required patch filter params raise an error."""
        patch_filter_config = create_patch_filter_configuration(
            patch_filter=patch_filter,
            patch_filter_params=patch_filter_params,
        )

        with pytest.raises(ValueError):
            create_ng_data_configuration(
                data_type="array",
                axes="YX",
                patch_size=(16, 16),
                batch_size=1,
                patch_filter=patch_filter_config,
            )
