import numpy as np
import pytest
import zarr
from numpy import array_equal

from careamics.config.data import NGDataConfig
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule


@pytest.fixture
def zarr_with_target_and_mask(tmp_path) -> str:
    """Three arrays in a single zarr group at the root."""
    path = tmp_path / "zarr_with_target_and_mask.zarr"
    zarr_file = zarr.create_group(path)
    assert path.exists()

    # arrays
    arrays = np.arange(3 * 16 * 16).reshape((3, 16, 16))
    targets = np.arange(3 * 16 * 16).reshape((3, 16, 16))
    masks = np.ones((3, 16, 16), dtype=bool)
    val = np.ones((16, 16))

    # exclude central frame
    masks[1] = np.zeros((16, 16), dtype=bool)

    # create groups
    input_group = zarr_file.create_group("input")
    target_group = zarr_file.create_group("target")
    mask_group = zarr_file.create_group("mask")
    val_group = zarr_file.create_group("val")

    # write arrays to zarr
    for i in range(arrays.shape[0]):
        input_group.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))
        target_group.create_array(f"array{i}", data=targets[i], chunks=(8, 8))
        mask_group.create_array(f"array{i}", data=masks[i], chunks=(8, 8))
    val_group.create_array("val_array", data=val, chunks=(8, 8))

    return path


def test_zarr_data_module(zarr_with_target_and_mask):

    # create uri
    g = zarr.open(zarr_with_target_and_mask)

    input_uris = str(g["input"].store_path)
    target_uris = str(g["target"].store_path)
    mask_uris = str(g["mask"].store_path)
    val_uris = str(g["val"].store_path)

    # basic config
    config = NGDataConfig(
        data_type="zarr",
        axes="YX",
        patching={
            "name": "random",
            "patch_size": (8, 8),
        },
        coord_filter={"name": "mask"},
        batch_size=1,
        seed=42,
        image_means=[0],
        image_stds=[1],
        target_means=[0],
        target_stds=[1],
    )

    datamodule = CareamicsDataModule(
        data_config=config,
        train_data=[input_uris],
        train_data_target=[target_uris],
        train_data_mask=[mask_uris],
        val_data=[val_uris],
    )
    # simulate training call
    datamodule.setup(stage="fit")

    # inspect train dataset
    train_dataset = datamodule.train_dataset
    assert len(train_dataset) > 0
    for i in range(len(train_dataset)):
        sample, target = train_dataset[i]

        # assess that the two image regions are equal (by design of the test zarr)
        assert array_equal(sample.data, target.data)

        # assert that it pulls from the correct mask
        source = train_dataset.coord_filter.mask_extractor.image_stacks[
            sample.region_spec["data_idx"]
        ].source
        assert source.split("/")[-1] == sample.source.split("/")[-1]
