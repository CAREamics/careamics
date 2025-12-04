import numpy as np
import pytest
import zarr

import careamics.lightning.dataset_ng.callbacks.prediction_writer as pd_writer
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_zarrs
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import (
    PatchSpecs,
    TileSpecs,
    TilingStrategy,
)

# to comply with ruff line length
WriteTilesZarr = pd_writer.write_tiles_zarr_strategy.WriteTilesZarr
_auto_chunks = pd_writer.write_tiles_zarr_strategy._auto_chunks
_update_data_shape = pd_writer.write_tiles_zarr_strategy._update_data_shape


def create_image_region(
    axes, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
) -> ImageRegionData:
    data_idx = patch_spec["data_idx"]
    source = extractor.image_stacks[data_idx].source
    shards = extractor.image_stacks[data_idx].shards
    chunks = extractor.image_stacks[data_idx].chunks
    return ImageRegionData(
        data=patch,
        source=str(source),
        dtype=str(extractor.image_stacks[data_idx].data_dtype),
        data_shape=extractor.image_stacks[data_idx].data_shape,
        axes=axes,
        region_spec=patch_spec,
        additional_metadata={
            "shards": str(shards),
            "chunks": str(chunks),
        },
    )


def gen_image_regions(
    axes: str, my_patch_extractor: PatchExtractor, my_strategy: TilingStrategy
):
    for i in range(my_strategy.n_patches):
        patch_spec: TileSpecs = my_strategy.get_patch_spec(i)
        patch = my_patch_extractor.extract_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )
        region = create_image_region(axes, patch, patch_spec, my_patch_extractor)

        yield region


@pytest.fixture
def zarr_path(tmp_path, shape, shards, chunks):
    array = np.random.rand(*shape).astype(np.float32)

    path = tmp_path / "test.zarr"
    g = zarr.create_group(path)
    array = g.create_array(
        name="image_stack",
        data=array,
        shards=shards,
        chunks=chunks,
    )

    return path


@pytest.mark.parametrize(
    "axes, data_shape, expected_shape",
    [
        ("YX", (1, 1, 32, 64), (32, 64)),
        ("ZYX", (1, 1, 8, 32, 64), (8, 32, 64)),
        ("YXZ", (1, 1, 8, 32, 64), (8, 32, 64)),
        ("CYX", (1, 3, 32, 64), (3, 32, 64)),
        ("YXC", (1, 3, 32, 64), (3, 32, 64)),
        ("CZYX", (1, 3, 32, 64, 64), (3, 32, 64, 64)),
        ("ZCYX", (1, 3, 32, 64, 64), (3, 32, 64, 64)),
        ("ZYXC", (1, 3, 32, 64, 64), (3, 32, 64, 64)),
        ("SCYX", (8, 3, 32, 64), (8, 3, 32, 64)),
        ("SYXC", (8, 3, 32, 64), (8, 3, 32, 64)),
        ("SCZYX", (8, 3, 32, 64, 64), (8, 3, 32, 64, 64)),
        ("SZCYX", (8, 3, 32, 64, 64), (8, 3, 32, 64, 64)),
        ("ZSYXC", (8, 3, 32, 64, 64), (8, 3, 32, 64, 64)),
    ],
)
def test_update_data_shape(axes, data_shape, expected_shape):
    new_shape = _update_data_shape(axes, data_shape)
    assert new_shape == expected_shape


@pytest.mark.parametrize(
    "axes, data_shape, expected_chunks",
    [
        # axes are original data, can be STCZYX in any order
        # data_shape is in format SC(Z)YX with potential singleton dimensions
        # expected_chunks is in format SC(Z)YX as data is currently not reshaped
        # simple usual shapes
        ("YX", (1, 1, 32, 64), (32, 64)),
        ("YX", (1, 1, 128, 32), (64, 32)),
        ("ZYX", (1, 1, 32, 64, 64), (32, 64, 64)),
        ("ZYX", (1, 1, 64, 128, 64), (64, 64, 64)),
        ("CYX", (1, 5, 64, 64), (1, 64, 64)),
        ("SYX", (5, 1, 64, 64), (1, 64, 64)),
        ("SCYX", (8, 5, 64, 64), (1, 1, 64, 64)),
        ("SCZYX", (5, 5, 32, 64, 64), (1, 1, 32, 64, 64)),
        # different orders (but YX together)
        ("YXZ", (1, 1, 32, 64, 64), (32, 64, 64)),
        ("YXC", (1, 5, 64, 64), (1, 64, 64)),
        ("SYXZ", (1, 1, 32, 64, 64), (1, 32, 64, 64)),
        ("CSYX", (8, 5, 64, 64), (1, 1, 64, 64)),
        ("SZCYX", (8, 5, 32, 64, 64), (1, 1, 32, 64, 64)),
    ],
)
def test_auto_chunks(axes, data_shape, expected_chunks):
    chunks = _auto_chunks(axes, data_shape)
    assert chunks == expected_chunks


@pytest.mark.parametrize(
    "axes, shape, shards, chunks",
    [
        ("YX", (32, 32), (16, 16), (8, 8)),
        ("ZYX", (16, 32, 32), (8, 16, 16), (4, 8, 8)),
        ("CYX", (3, 32, 32), (1, 16, 16), (1, 8, 8)),
        ("CZYX", (3, 16, 32, 32), (1, 8, 16, 16), (1, 4, 8, 8)),
        ("SCYX", (2, 3, 32, 32), (1, 1, 16, 16), (1, 1, 8, 8)),
        ("SCZYX", (2, 3, 16, 32, 32), (1, 1, 8, 16, 16), (1, 1, 4, 8, 8)),
    ],
)
def test_write_tile_identity(tmp_path, zarr_path, axes):
    """Test that `write_tile` correctly writes the data.

    No need to test with different axes order since the data coming to the writer
    is always in C(Z)YX format, with potential singleton dimensions.
    """
    # create extractor and tiling strategy
    image_stacks = load_zarrs(
        source=[str(zarr_path)],
        axes=axes,
    )
    patch_extractor = PatchExtractor(image_stacks)

    strategy = TilingStrategy(
        data_shapes=[image_stacks[0].data_shape],
        patch_size=(8, 8) if "Z" not in axes else (4, 8, 8),
        overlaps=(4, 4) if "Z" not in axes else (2, 4, 4),
    )

    # use writer to write predictions
    writer = WriteTilesZarr()
    for region in gen_image_regions(axes, patch_extractor, strategy):
        writer.write_tile(tmp_path, region)

    # check that the array has been writtent correctly
    assert (tmp_path / "test_output.zarr").exists()

    g_output = zarr.open_group(tmp_path / "test_output.zarr", mode="r")

    # group["array"][:] forces loading the full array into memory
    assert np.array_equal(g_output["image_stack"][:], image_stacks[0]._array[:])
    assert g_output["image_stack"].shards == image_stacks[0].shards
    assert g_output["image_stack"].chunks == image_stacks[0].chunks


# TODO test chunking and sharding errors and handling (e.g. when missing)
# TODO test that different chunk sizes are handled correctly
# TODO test writing different arrays, and in different groups and zarr files


# TODO refactor this test to make simpler
def test_zarr_prediction_callback_identity(tmp_path):
    """Test writing multiple arrays in a different hierarchy levels."""
    # create data
    arrays = np.arange(6 * 5 * 32 * 32).reshape((6, 5, 32, 32))
    shards = (1, 16, 16)
    chunks = (1, 8, 8)
    axes = "SYX"

    # write zarr sources to two different zarrs, at different levels
    path = tmp_path / "source.zarr"
    g = zarr.create_group(path)

    image1_group = g.create_group("images1")
    single_array = image1_group.create_array(
        name="single_image",
        data=arrays[0],
        shards=shards,
        chunks=chunks,
    )
    array_uris = [single_array.store_path]  # uris to the arrays

    image2_group = g.create_group("images2")
    for i in range(1, 5):
        array = image2_group.create_array(
            name=f"image_stack_{i}",
            data=arrays[i],
            shards=shards,
            chunks=chunks,
        )
        array_uris.append(array.store_path)

    path2 = tmp_path / "source2.zarr"
    g2 = zarr.create_group(path2)
    array_root = g2.create_array(
        name="root_array",
        data=arrays[5],
        shards=shards,
        chunks=chunks,
    )
    array_uris.append(array_root.store_path)

    # create extractor and tiling strategy
    image_stacks = load_zarrs(
        source=array_uris,
        axes=axes,
    )
    patch_extractor = PatchExtractor(image_stacks)

    strategy = TilingStrategy(
        data_shapes=[(5, 1, 32, 32) for _ in range(len(array_uris))],
        patch_size=(8, 8),
        overlaps=(4, 4),
    )
    assert strategy.n_patches == 6 * 5 * ((32 - 4) / (8 - 4)) ** 2

    # use writer to write predictions
    writer = WriteTilesZarr()
    for region in gen_image_regions(axes, patch_extractor, strategy):
        writer.write_tile(tmp_path, region)

    # check that the arrays have been writtent correctly to the first zarr
    assert (tmp_path / "source_output.zarr").exists()

    g_output = zarr.open_group(tmp_path / "source_output.zarr", mode="r")
    assert np.array_equal(g_output["images1/single_image"][:], arrays[0])
    assert g_output["images1/single_image"].shards == shards
    assert g_output["images1/single_image"].chunks == chunks
    for i in range(1, 5):
        assert np.array_equal(g_output[f"images2/image_stack_{i}"][:], arrays[i])
        assert g_output[f"images2/image_stack_{i}"].shards == shards
        assert g_output[f"images2/image_stack_{i}"].chunks == chunks

    # check that the array has been written correctly to the second zarr
    assert (tmp_path / "source2_output.zarr").exists()
    g_output2 = zarr.open_group(tmp_path / "source2_output.zarr", mode="r")
    assert np.array_equal(g_output2["root_array"][:], arrays[5])
    assert g_output2["root_array"].shards == shards
    assert g_output2["root_array"].chunks == chunks
