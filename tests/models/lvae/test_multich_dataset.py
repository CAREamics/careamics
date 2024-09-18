import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.lvae_training.dataset import (
    DatasetConfig,
    DataSplitType,
    DataType,
    LCMultiChDloader,
    MultiChDloader,
)
from careamics.lvae_training.dataset.utils.data_utils import (
    get_datasplit_tuples,
    load_tiff,
)

pytestmark = pytest.mark.lvae


def load_data_fn_example(
    data_config: DatasetConfig,
    fpath: str,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    fpath1 = os.path.join(fpath, data_config.ch1_fname)
    fpath2 = os.path.join(fpath, data_config.ch2_fname)
    fpaths = [fpath1, fpath2]

    if "ch_input_fname" in data_config:
        fpath0 = os.path.join(fpath, data_config.ch_input_fname)
        fpaths = [fpath0] + fpaths

    data = np.concatenate([load_tiff(fpath)[..., None] for fpath in fpaths], axis=3)
    if data_config.data_type == DataType.PredictedTiffData:
        assert len(data.shape) == 5 and data.shape[-1] == 1
        data = data[..., 0].copy()

    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data), starting_test=True
    )
    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)


@pytest.fixture
def dummy_data_path(tmp_path: Path) -> str:
    max_val = 65535

    example_data_ch1 = np.random.rand(55, 512, 512)
    example_data_ch1 = example_data_ch1 * max_val
    example_data_ch1 = example_data_ch1.astype(np.uint16)

    example_data_ch2 = np.random.rand(55, 512, 512)
    example_data_ch2 = example_data_ch2 * max_val
    example_data_ch2 = example_data_ch2.astype(np.uint16)

    tifffile.imwrite(tmp_path / "ch1.tiff", example_data_ch1)
    tifffile.imwrite(tmp_path / "ch2.tiff", example_data_ch2)

    return str(tmp_path)


@pytest.fixture
def default_config() -> DatasetConfig:
    return DatasetConfig(
        ch1_fname="ch1.tiff",
        ch2_fname="ch2.tiff",
        # TODO: something breaks when set to ALL
        datasplit_type=DataSplitType.Train,
        data_type=DataType.SeparateTiffData,
        enable_gaussian_noise=False,
        image_size=128,
        input_has_dependant_noise=True,
        multiscale_lowres_count=None,
        num_channels=2,
        enable_random_cropping=False,
        enable_rotation_aug=False,
    )


def test_create_vae_dataset(default_config, dummy_data_path):
    dataset = MultiChDloader(
        default_config,
        dummy_data_path,
        load_data_fn=load_data_fn_example,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    max_val = dataset.get_max_val()
    assert max_val is not None, max_val > 0

    mean_val, std_val = dataset.compute_mean_std()
    dataset.set_mean_std(mean_val, std_val)

    sample = dataset[0]
    assert len(sample) == 2

    inputs, targets = sample
    assert inputs.shape == (1, 128, 128)
    assert len(targets) == 2

    for channel in targets:
        assert channel.shape == (128, 128)

    # input and target are normalized
    assert inputs.mean() < 1
    assert inputs.std() < 1.1
    assert targets[0].mean() < 1
    assert targets[0].std() < 1.1


@pytest.mark.parametrize("num_scales", [1, 2, 3])
def test_create_lc_dataset(default_config, dummy_data_path, num_scales: int):
    lc_config = DatasetConfig(**default_config.model_dump(exclude_none=True))
    lc_config.multiscale_lowres_count = num_scales

    padding_kwargs = {"mode": "reflect"}
    lc_config.padding_kwargs = padding_kwargs
    lc_config.overlapping_padding_kwargs = padding_kwargs

    dataset = LCMultiChDloader(
        lc_config,
        dummy_data_path,
        load_data_fn=load_data_fn_example,
        val_fraction=0.1,
        test_fraction=0.1,
    )

    max_val = dataset.get_max_val()
    assert max_val is not None, max_val > 0

    mean_val, std_val = dataset.compute_mean_std()

    dataset.set_mean_std(mean_val, std_val)

    sample = dataset[0]
    assert len(sample) == 2

    inputs, targets = sample
    assert inputs.shape == (num_scales, 128, 128)
    assert len(targets) == 2

    for channel in targets:
        assert channel.shape == (128, 128)

    # input and target are normalized
    assert inputs.mean() < 1
    assert inputs.std() < 1.1
    assert targets[0].mean() < 1
    assert targets[0].std() < 1.1
