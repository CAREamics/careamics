from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import pytest
import tifffile

from careamics.config.data import NGDataConfig
from careamics.config.support import SupportedData
from careamics.dataset_ng.dataset import CareamicsDataset
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.image_stack import FileImageStack
from careamics.dataset_ng.patch_extractor.limit_file_extractor import (
    LimitFilesPatchExtractor,
)
from careamics.lightning.dataset_ng.data_module import (
    CareamicsDataModule,
    _PredData,
    _TrainVal,
    _TrainValSplit,
    _validate_data,
)

# TODO add tests for the various types, for mismatching input/target/mask lengths, etc.
# TODO add tests for validation and prediction modes. can we use a single
# datamodule for all 3 modes?


def _patch_file_image_stacks(dataset: CareamicsDataset[FileImageStack]):
    mocks = []
    for image_stack in dataset.input_extractor.image_stacks:
        load_patch = patch.object(image_stack, "load", wraps=image_stack.load)
        close_patch = patch.object(image_stack, "close", wraps=image_stack.close)

        load_mock = load_patch.start()
        close_mock = close_patch.start()
        mocks.append((load_mock, close_mock))

    return mocks


def test_not_in_mem_tiff(tmp_path: Path):
    """
    Test that the correct `Sampler` and `PatchExtractor` are set for not in memory
    tiffs. Test that the FileImageStack load and close methods are only called once.
    """

    # set up test data
    data_shapes = [(64, 48), (55, 54), (71, 65), (32, 32)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    paths = [tmp_path / f"{i}.tiff" for i in range(len(data_shapes))]
    for path, data in zip(paths, input_data, strict=True):
        tifffile.imwrite(path, data, metadata={"axes": "YX"})

    # basic config
    config = NGDataConfig(
        mode="training",
        data_type="tiff",
        axes="YX",
        patching={
            "name": "random",
            "patch_size": (16, 16),
        },
        batch_size=4,
        in_memory=False,
        seed=42,
        normalization={
            "name": "mean_std",
            "input_means": [0],
            "input_stds": [1],
        },
    )

    datamodule = CareamicsDataModule(
        data_config=config,
        train_data=paths[:-1],
        val_data=[paths[-1]],
    )
    # simulate training call
    datamodule.setup(stage="fit")
    train_files_mock_funcs = _patch_file_image_stacks(datamodule.train_dataset)
    val_files_mock_funcs = _patch_file_image_stacks(datamodule.val_dataset)
    dataloaders = [datamodule.train_dataloader(), datamodule.val_dataloader()]

    # test that the sampler and extractor types are correct
    for dataloader in dataloaders:
        assert isinstance(dataloader.sampler, GroupedIndexSampler)
        dataset = cast(CareamicsDataset, dataloader.dataset)
        assert isinstance(dataset.input_extractor, LimitFilesPatchExtractor)
        # to make sure load and closed will be called only once for each FileImageStack
        for _ in dataloader:
            pass

    # NOTE: close will not be called if the image stack was the last one
    for load_mock, close_mock in train_files_mock_funcs:
        load_mock.assert_called_once()
        assert close_mock.call_count <= 1

    for load_mock, close_mock in val_files_mock_funcs:
        load_mock.assert_called_once()
        assert close_mock.call_count <= 1


@pytest.mark.parametrize(
    ["in_memory", "correct_sampler"], [(False, GroupedIndexSampler), (True, None)]
)
def test_sampler(tmp_path: Path, in_memory, correct_sampler):
    """Test `_sampler` method returns the correct sampler for"""

    # set up test data
    data_shapes = [(64, 48), (55, 54), (71, 65), (32, 32)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    paths = [tmp_path / f"{i}.tiff" for i in range(len(data_shapes))]
    for path, data in zip(paths, input_data, strict=True):
        tifffile.imwrite(path, data, metadata={"axes": "YX"})

    # basic config
    config = NGDataConfig(
        mode="training",
        data_type="tiff",
        axes="YX",
        patching={
            "name": "random",
            "patch_size": (16, 16),
        },
        batch_size=4,
        in_memory=in_memory,
        normalization={
            "name": "mean_std",
            "input_means": [0],
            "input_stds": [1],
        },
    )

    datamodule = CareamicsDataModule(
        data_config=config,
        train_data=paths[:-1],
        val_data=[paths[-1]],
    )
    datamodule.setup("fit")

    datasets: list[Literal["train", "val"]] = ["train", "val"]
    for ds in datasets:
        sampler = datamodule._sampler(ds)
        # if not in memory and tiff should be GroupedIndexSampler
        # if in memory should be None
        if correct_sampler is not None:
            assert isinstance(sampler, correct_sampler)
        else:
            assert sampler is None


@pytest.mark.parametrize(
    "train_data, val_data, val_split, pred_data, output",
    [
        [True, True, False, False, _TrainVal],
        [True, False, True, False, _TrainValSplit],
        [False, False, False, True, _PredData],
        [True, True, True, False, None],
        [True, False, False, True, None],
        [True, True, True, True, None],
        [False, False, False, False, None],
    ],
    ids=[
        "train-val",
        "train-val-split",
        "pred",
        "train-val-and-split-error",
        "train-pred-error",
        "all-error",
        "none-error",
    ],
)
def test_validate_data(
    train_data: bool,
    val_data: bool,
    val_split: bool,
    pred_data: bool,
    # if the combination should fail output is set to None
    output: type[_TrainVal] | type[_TrainValSplit] | type[_PredData] | None,
):
    """
    Test different combinations of data for the data module are validated correctly.
    """
    train_input = np.ones((64, 64)) if train_data else None
    val_input = np.ones((32, 32)) if val_data else None
    val_percentage = 0.1 if val_split else None
    pred_input = np.ones((96, 96)) if pred_data else None

    if output is not None:
        data = _validate_data(
            data_type=SupportedData.ARRAY,
            train_data=train_input,
            val_data=val_input,
            val_percentage=val_percentage,
            pred_data=pred_input,
        )
        assert isinstance(data, output)
    else:
        with pytest.raises(ValueError):
            _ = _validate_data(
                data_type=SupportedData.ARRAY,
                train_data=train_input,
                val_data=val_input,
                val_percentage=val_percentage,
                pred_data=pred_input,
            )
