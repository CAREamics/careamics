from pathlib import Path

import numpy as np
import tifffile

from careamics.lvae_training.dataset.configs.vae_data_config import (
    DataSplitType,
    DataType,
    VaeDatasetConfig,
)
from careamics.lvae_training.dataset.multifile_dataset import MultiFileDset


def random_uint16_data(shape, max_value):
    data = np.random.rand(*shape)
    data = data * max_value
    data = data.astype(np.uint16)
    return data


def test_create_vae_dataset(tmp_path: Path, num_files=3):
    for i in range(num_files):
        example_data = random_uint16_data((25, 512, 512, 3), max_value=65535)
        tifffile.imwrite(tmp_path / f"{i}.tif", example_data)

    config = VaeDatasetConfig(
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

    dataset = MultiFileDset(config, tmp_path, val_fraction=0.1, test_fraction=0.1)

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
