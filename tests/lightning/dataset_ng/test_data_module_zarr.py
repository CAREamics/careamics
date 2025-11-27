import zarr
from numpy import array_equal

from careamics.config.data import NGDataConfig
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule


def test_zarr_data_module(zarr_with_target_and_mask):
    assert zarr_with_target_and_mask.exists()

    # create uri
    g = zarr.open(zarr_with_target_and_mask)
    assert "input" in g

    input_uris = str(g["input"].store_path)
    target_uris = str(g["target"].store_path)
    mask_uris = str(g["mask"].store_path)
    val_uris = str(g["val"].store_path)

    # basic config
    config = NGDataConfig(
        mode="training",
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
