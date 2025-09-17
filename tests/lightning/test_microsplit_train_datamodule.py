"""Tests for MicroSplitDataModule data statistics handling."""

import numpy as np
import tifffile

from careamics.lightning.microsplit_data_module import (
    MicroSplitDataModule,
    get_train_val_data,
)
from careamics.lvae_training.dataset import (
    DataSplitType,
    DataType,
    MicroSplitDataConfig,
)
from careamics.lvae_training.dataset.types import TilingMode


def test_microsplit_datamodule_data_stats(tmp_path):
    rng = np.random.default_rng(42)

    # Create channel directories
    ch0_dir = tmp_path / "ch0"
    ch1_dir = tmp_path / "ch1"
    ch0_dir.mkdir()
    ch1_dir.mkdir()

    # Create random data for each channel
    for i in range(5):
        data_ch0 = rng.integers(0, 255, (64, 64)).astype(np.float32)
        data_ch1 = rng.integers(0, 255, (64, 64)).astype(np.float32)

        tifffile.imwrite(ch0_dir / f"img_{i}.tif", data_ch0)
        tifffile.imwrite(ch1_dir / f"img_{i}.tif", data_ch1)

    config = MicroSplitDataConfig(
        data_type=DataType.HTLIF24Data,
        image_size=(64, 64),
        grid_size=32,
        num_channels=2,
        batch_size=2,
        datasplit_type=DataSplitType.Train,
        multiscale_lowres_count=1,
        tiling_mode=TilingMode.ShiftBoundary,
        train_dataloader_params={},
        val_dataloader_params={},
    )

    data_module = MicroSplitDataModule(
        data_config=config,
        train_data=str(tmp_path),
        read_source_func=get_train_val_data,
    )

    assert data_module.data_stats is not None
    assert len(data_module.data_stats) == 2

    mean_dict, std_dict = data_module.data_stats
    assert isinstance(mean_dict, dict)
    assert isinstance(std_dict, dict)
    assert "input" in mean_dict
    assert "input" in std_dict
