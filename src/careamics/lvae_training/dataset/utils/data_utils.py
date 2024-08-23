"""
Utility functions needed by dataloader & co.
"""

import os
from typing import List

import numpy as np
from skimage.io import imread, imsave

from careamics.lvae_training.dataset.configs.vae_data_config import (
    DataSplitType,
    DataType,
)

Multifile_Datasets = [
    "Pavia3SeqData",
    "TavernaSox2GolgiV2",
    "Dao3ChannelWithInput",
    "ExpMicroscopyV2",
    "Dao3Channel",
    "TavernaSox2Golgi",
    "ExpMicroscopyV1",
    "OptiMEM100_014",
]

Singlefile_Datasets = ["Elisa3DData", "NicolaData", "HTIba1Ki67", "SeparateTiffData"]


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")
    return data


def save_tiff(path, data):
    imsave(path, data, plugin="tifffile")


def load_tiffs(paths):
    data = [load_tiff(path) for path in paths]
    return np.concatenate(data, axis=0)


def split_in_half(s, e):
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def adjust_for_imbalance_in_fraction_value(
    val: List[int],
    test: List[int],
    val_fraction: float,
    test_fraction: float,
    total_size: int,
):
    """
    here, val and test are divided almost equally. Here, we need to take into account their respective fractions
    and pick elements rendomly from one array and put in the other array.
    """
    if val_fraction == 0:
        test += val
        val = []
    elif test_fraction == 0:
        val += test
        test = []
    else:
        diff_fraction = test_fraction - val_fraction
        if diff_fraction > 0:
            imb_count = int(diff_fraction * total_size / 2)
            val = list(np.random.RandomState(seed=955).permutation(val))
            test += val[:imb_count]
            val = val[imb_count:]
        elif diff_fraction < 0:
            imb_count = int(-1 * diff_fraction * total_size / 2)
            test = list(np.random.RandomState(seed=955).permutation(test))
            val += test[:imb_count]
            test = test[imb_count:]
    return val, test


def get_train_val_data(
    data_config,
    fpath,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    allow_generation=False,  # TODO: what is this
):
    """
    Load the data from the given path and split them in training, validation and test sets.

    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type.name in Singlefile_Datasets:
        fpath1 = os.path.join(fpath, data_config.ch1_fname)
        fpath2 = os.path.join(fpath, data_config.ch2_fname)
        fpaths = [fpath1, fpath2]
        fpath0 = ""
        if "ch_input_fname" in data_config:
            fpath0 = os.path.join(fpath, data_config.ch_input_fname)
            fpaths = [fpath0] + fpaths

        print(
            f"Loading from {fpath} Channels: "
            f"{fpath1},{fpath2}, inp:{fpath0} Mode:{data_config.data_type}"
        )

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

    elif data_config.data_type.name in Multifile_Datasets:
        files = fpath.glob("*.tif*")
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

    else:
        raise TypeError(f"Data type {data_config.data_type} not supported")


def get_datasplit_tuples(
    val_fraction: float,
    test_fraction: float,
    total_size: int,
    starting_test: bool = False,
):
    if starting_test:
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))
        val = list(range(test[-1] + 1, test[-1] + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))
    else:
        # {test,val}=> train
        test_val_size = int((val_fraction + test_fraction) * total_size)
        train = list(range(test_val_size, total_size))

        if test_val_size == 0:
            test = []
            val = []
            return train, val, test

        # Split the test and validation in chunks.
        chunksize = max(1, min(3, test_val_size // 2))

        nchunks = test_val_size // chunksize

        test = []
        val = []
        s = 0
        for i in range(nchunks):
            if i % 2 == 0:
                val += list(np.arange(s, s + chunksize))
            else:
                test += list(np.arange(s, s + chunksize))
            s += chunksize

        if i % 2 == 0:
            test += list(np.arange(s, test_val_size))
        else:
            p1, p2 = split_in_half(s, test_val_size)
            test += p1
            val += p2

    val, test = adjust_for_imbalance_in_fraction_value(
        val, test, val_fraction, test_fraction, total_size
    )

    return train, val, test
