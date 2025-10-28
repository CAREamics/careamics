from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import tifffile

from careamics.config.data import NGDataConfig
from careamics.dataset_ng.dataset import CareamicsDataset
from careamics.dataset_ng.grouped_index_sampler import GroupedIndexSampler
from careamics.dataset_ng.patch_extractor.limit_file_extractor import (
    LimitFilesPatchExtractor,
)
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule


def test_not_in_mem_tiff(tmp_path: Path):
    """
    Test that the correct `Sampler` and `PatchExtractor` are set for not in memory
    tiffs.
    """

    # set up test data
    data_shapes = [(64, 48), (55, 54), (71, 65), (32, 32)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    paths = [tmp_path / f"{i}.tiff" for i in range(len(data_shapes))]
    for path, data in zip(paths, input_data, strict=True):
        tifffile.imwrite(path, data, metadata={"axes": "YX"})

    # basic config
    config = NGDataConfig(
        data_type="tiff",
        axes="YX",
        patching={
            "name": "random",
            "patch_size": (16, 16),
        },
        batch_size=4,
    )

    datamodule = CareamicsDataModule(
        data_config=config,
        train_data=paths[:-1],
        val_data=[paths[-1]],
        use_in_memory=False,
    )
    # simulate training call
    datamodule.setup(stage="fit")
    dataloaders = [datamodule.train_dataloader(), datamodule.val_dataloader()]

    # just test that the sampler and extractor types are correct
    # functionality of the components have their own unit tests
    for dataloader in dataloaders:
        assert isinstance(dataloader.sampler, GroupedIndexSampler)
        dataset = cast(CareamicsDataset, dataloader.dataset)
        assert isinstance(dataset.input_extractor, LimitFilesPatchExtractor)


@pytest.mark.parametrize(
    ["in_memory", "correct_sampler"], [(False, GroupedIndexSampler), (True, None)]
)
def test_sampler(tmp_path: Path, in_memory, correct_sampler):
    """Test `_sampler` method returns the correct sampler for"""

    tmp_path = Path("data")

    # set up test data
    data_shapes = [(64, 48), (55, 54), (71, 65), (32, 32)]
    input_data = [np.arange(np.prod(shape)).reshape(shape) for shape in data_shapes]
    paths = [tmp_path / f"{i}.tiff" for i in range(len(data_shapes))]
    for path, data in zip(paths, input_data, strict=True):
        tifffile.imwrite(path, data, metadata={"axes": "YX"})

    # basic config
    config = NGDataConfig(
        data_type="tiff",
        axes="YX",
        patching={
            "name": "random",
            "patch_size": (16, 16),
        },
        batch_size=4,
    )

    datamodule = CareamicsDataModule(
        data_config=config,
        train_data=paths[:-1],
        val_data=[paths[-1]],
        use_in_memory=in_memory,
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
