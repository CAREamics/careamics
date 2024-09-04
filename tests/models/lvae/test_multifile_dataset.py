from pathlib import Path
from typing import Union

import numpy as np
import tifffile

from careamics.lvae_training.dataset.configs.lc_dataset_config import LCDatasetConfig
from careamics.lvae_training.dataset.configs.multich_data_config import (
    DataSplitType,
    DataType,
    MultiChDatasetConfig,
)
from careamics.lvae_training.dataset.multifile_dataset import MultiFileDset
from careamics.lvae_training.dataset.utils.data_utils import (
    get_datasplit_tuples,
    load_tiff,
)


def random_uint16_data(shape, max_value):
    data = np.random.rand(*shape)
    data = data * max_value
    data = data.astype(np.uint16)
    return data


def load_data_fn_example(
    data_config: Union[MultiChDatasetConfig, LCDatasetConfig],
    fpath: str,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    files = Path(fpath).glob("*.tif*")
    data = [load_tiff(fpath) for fpath in files]

    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data), starting_test=False
    )
    if datasplit_type == DataSplitType.Train:
        return np.take(data, train_idx, axis=0).astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return np.take(data, val_idx, axis=0).astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return np.take(data, test_idx, axis=0).astype(np.float32)


def test_create_vae_dataset(tmp_path: Path, num_files=3):
    for i in range(num_files):
        example_data = random_uint16_data((25, 512, 512, 3), max_value=65535)
        tifffile.imwrite(tmp_path / f"{i}.tif", example_data)

    config = MultiChDatasetConfig(
        image_size=64,
        num_channels=3,
        input_idx=2,
        target_idx_list=[0, 1],
        datasplit_type=DataSplitType.Train,
        data_type=DataType.Pavia3SeqData,
        enable_gaussian_noise=False,
        input_has_dependant_noise=True,
        multiscale_lowres_count=None,
        enable_random_cropping=False,
        enable_rotation_aug=False,
    )

    dataset = MultiFileDset(
        config,
        tmp_path,
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
    assert inputs.shape == (1, 64, 64)
    assert len(targets) == 2

    for channel in targets:
        assert channel.shape == (64, 64)

    # input is normalized
    assert inputs.mean() < 1
    assert inputs.std() < 1.1

    # output is not normalized
    assert targets[0].mean() > 1
    assert targets[0].std() > 1
