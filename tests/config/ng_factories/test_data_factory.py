import os
import sys

import pytest

from careamics.config.augmentations import (
    N2VManipulateConfig,
    XYFlipConfig,
    XYRandomRotate90Config,
)
from careamics.config.data import NGDataConfig
from careamics.config.data.ng_data_config import get_default_num_workers
from careamics.config.data.patching_strategies import StratifiedPatchingConfig
from careamics.config.ng_factories.data_factory import (
    create_ng_data_configuration,
    list_spatial_augmentations,
)
from careamics.config.support import SupportedTransform


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


class TestNGDataConfiguration:

    def test_default_aug(self):
        """Test that the default augmentations are present in the configuration."""
        config: NGDataConfig = create_ng_data_configuration(
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

        assert isinstance(config.patching, StratifiedPatchingConfig)

    def test_num_workers_explicit(self):
        """Test that an explicit num_workers value is passed through unchanged."""
        config: NGDataConfig = create_ng_data_configuration(
            data_type="array",
            axes="YX",
            patch_size=(16, 16),
            batch_size=1,
            num_workers=4,
        )

        assert config.train_dataloader_params["num_workers"] == 4
        assert config.val_dataloader_params["num_workers"] == 4
        assert config.pred_dataloader_params["num_workers"] == 4

    def test_get_default_num_workers_in_pytest(self):
        """Test that get_default_num_workers returns 0 when running under pytest."""
        assert get_default_num_workers() == 0

    def test_get_default_num_workers_linux(self, monkeypatch: pytest.MonkeyPatch):
        """Test that Linux returns min(cpu_count - 1, 4)."""
        expected = min((os.cpu_count() or 1) - 1, 4)
        monkeypatch.setattr(
            "careamics.config.data.ng_data_config.platform.system", lambda: "Linux"
        )
        monkeypatch.delitem(sys.modules, "pytest")
        assert get_default_num_workers() == expected

    def test_get_default_num_workers_windows_returns_0(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that Windows returns 0."""
        monkeypatch.setattr(
            "careamics.config.data.ng_data_config.platform.system", lambda: "Windows"
        )
        monkeypatch.delitem(sys.modules, "pytest")
        assert get_default_num_workers() == 0

    def test_get_default_num_workers_intel_mac_returns_0(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that macOS without MPS (Intel Mac) returns 0."""
        import torch

        monkeypatch.setattr(
            "careamics.config.data.ng_data_config.platform.system", lambda: "Darwin"
        )
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.delitem(sys.modules, "pytest")
        assert get_default_num_workers() == 0
