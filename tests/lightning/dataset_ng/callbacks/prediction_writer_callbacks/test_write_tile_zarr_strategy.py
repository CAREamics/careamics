from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr
from numpy.typing import NDArray

import careamics.lightning.dataset_ng.callbacks.prediction_writer as pd_writer
from careamics.config.data import NGDataConfig
from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset_ng.dataset import CareamicsDataset, ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_arrays, load_tiffs, load_zarrs
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

# TODO test chunking and sharding errors and handling (e.g. when missing)


def create_image_region(
    axes, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
) -> ImageRegionData:
    data_idx = patch_spec["data_idx"]
    source = extractor.image_stacks[data_idx].source
    shards = (
        extractor.image_stacks[data_idx].shards
        if hasattr(extractor.image_stacks[data_idx], "shards")
        else None
    )
    chunks = (
        extractor.image_stacks[data_idx].chunks
        if hasattr(extractor.image_stacks[data_idx], "chunks")
        else None
    )
    return ImageRegionData(
        data=patch,
        source=str(source),
        dtype=str(extractor.image_stacks[data_idx].data_dtype),
        data_shape=extractor.image_stacks[data_idx].data_shape,
        axes=axes,
        region_spec=patch_spec,
        additional_metadata={
            "shards": shards,
            "chunks": chunks,
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
def data_config(axes, shape, channels) -> NGDataConfig:
    # create tiling strategy
    if "Z" in axes:
        tile_size = (8, 16, 16)
        overlaps = (2, 4, 4)
    else:
        tile_size = (16, 16)
        overlaps = (4, 4)

    if "C" in axes:
        n_channels = shape[axes.index("C")]

        if channels is not None:
            n_channels = len(channels)
    else:
        n_channels = 1

    return NGDataConfig(
        mode="predicting",
        data_type="zarr",
        patching={
            "name": "tiled",
            "patch_size": tile_size,
            "overlaps": overlaps,
        },
        axes=axes,
        channels=channels,
        image_means=[0.0 for _ in range(n_channels)],
        image_stds=[1.0 for _ in range(n_channels)],
    )


# TODO this is very similar to the fixture in test_ng_stitch_prediction.py, refactor
@pytest.fixture
def tiles(
    tmp_path, data_config: NGDataConfig, n_data, shape, shards, chunks
) -> tuple[NDArray, list[ImageRegionData]]:
    """Create tiles.

    Note that the tiles will be slightly different because the dataset changes the dtype
    of the data and performs normalization.

    We must use np.testing.assert_allclose(stitched_array, array, rtol=1e-5, atol=0),
    with relative tolerance as the errors scale with the values and we use
    np.arange to create the data.

    Returns
    -------
    np.ndarray
        Original array of shape DSC(Z)YX, where D is data.
    list of ImageRegionData
        Extracted tiles.
    """
    # create data
    array = np.arange(n_data * np.prod(shape)).reshape((n_data, *shape))

    sources = []
    root = tmp_path / "input_data"
    root.mkdir(parents=True, exist_ok=True)

    even = root / "even.zarr"
    odd = root / "odd.zarr"

    for i in range(n_data):
        if i % 2 == 0:
            pth = even
        else:
            pth = odd

        if not pth.exists():
            g = zarr.create_group(pth)
        else:
            g = zarr.open(pth, mode="a")

        # write array
        arr = g.create_array(
            name=f"array_{i}",
            data=array[i],
            shards=shards,
            chunks=chunks,
        )
        sources.append(arr.store_path)

    if "S" in data_config.axes:
        if "C" in data_config.axes:
            shape_with_sc = shape
        else:
            shape_with_sc = (shape[0], 1, *shape[1:])
    else:
        if "C" in data_config.axes:
            shape_with_sc = (1, *shape)
        else:
            shape_with_sc = (1, 1, *shape)

    tiling_strategy = TilingStrategy(
        data_shapes=[shape_with_sc] * n_data,
        patch_size=data_config.patching.patch_size,
        overlaps=data_config.patching.overlaps,
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    image_stacks = load_zarrs(source=sources, axes=data_config.axes)
    patch_extractor = PatchExtractor(image_stacks)

    # create dataset
    dataset = CareamicsDataset(
        data_config=data_config,
        input_extractor=patch_extractor,
    )

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tiles.append(dataset[i][0])

    # reshape array for testing
    arrays = [reshape_array(array[i], data_config.axes) for i in range(array.shape[0])]

    return np.stack(arrays, axis=0), tiles


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
        ("YX", (1, 1, 128, 32), (128, 32)),
        ("ZYX", (1, 1, 32, 64, 64), (32, 64, 64)),
        ("ZYX", (1, 1, 64, 128, 64), (64, 128, 64)),
        ("CYX", (1, 5, 64, 64), (1, 64, 64)),
        ("SYX", (5, 1, 64, 256), (1, 64, 128)),
        ("SCYX", (8, 5, 64, 64), (1, 1, 64, 64)),
        ("SCZYX", (5, 5, 32, 256, 64), (1, 1, 32, 128, 64)),
        # different orders (but YX together)
        ("YXZ", (1, 1, 32, 64, 64), (32, 64, 64)),
        ("YXC", (1, 5, 64, 64), (1, 64, 64)),
        ("SYXZ", (1, 1, 32, 64, 64), (1, 32, 64, 64)),
        ("CSYX", (8, 5, 64, 64), (1, 1, 64, 64)),
        ("SZCYX", (8, 5, 512, 256, 64), (1, 1, 128, 128, 64)),
        # T dimension
        ("TYX", (5, 1, 64, 64), (1, 64, 64)),
        ("TCYX", (5, 3, 64, 64), (1, 1, 64, 64)),
        ("STYX", (5, 1, 64, 64), (1, 64, 64)),  # S and T together
        ("STCYX", (5, 3, 256, 64), (1, 1, 128, 64)),
    ],
)
def test_auto_chunks(axes, data_shape, expected_chunks):
    chunks = _auto_chunks(axes, data_shape)
    assert chunks == expected_chunks


@pytest.mark.parametrize("n_data", [1, 3])
@pytest.mark.parametrize(
    "axes, shape, shards, chunks, channels",
    [
        ("YX", (32, 32), (16, 16), (8, 8), None),
        ("CYX", (3, 32, 32), (1, 16, 16), (1, 8, 8), None),
        ("CYX", (3, 32, 32), (1, 16, 16), (1, 8, 8), [1]),
        ("CYX", (3, 32, 32), (1, 16, 16), (1, 8, 8), [0, 2]),
        ("ZYX", (16, 32, 32), (8, 16, 16), (4, 8, 8), None),
        ("CZYX", (3, 16, 32, 32), (1, 8, 16, 16), (1, 4, 8, 8), None),
        ("CZYX", (3, 16, 32, 32), (1, 8, 16, 16), (1, 4, 8, 8), [1]),
        ("CZYX", (3, 16, 32, 32), (1, 8, 16, 16), (1, 4, 8, 8), [0, 2]),
        ("SZYX", (5, 16, 32, 32), (1, 8, 16, 16), (1, 4, 8, 8), None),
        ("SCZYX", (5, 3, 16, 32, 32), (1, 1, 8, 16, 16), (1, 1, 4, 8, 8), None),
        ("SCZYX", (5, 3, 16, 32, 32), (1, 1, 8, 16, 16), (1, 1, 4, 8, 8), [1]),
        ("SCZYX", (5, 3, 16, 32, 32), (1, 1, 8, 16, 16), (1, 1, 4, 8, 8), [0, 2]),
    ],
)
def test_write_tile_identity(tmp_path, tiles, axes, shards, chunks, channels):
    """Test that `write_tile` correctly writes the data.

    No need to test with different axes order since the data coming to the writer
    is always in C(Z)YX format, with potential singleton dimensions.
    """
    arrays, tiles_list = tiles

    source_set = {tile.source for tile in tiles_list}

    # use writer to write predictions
    writer = WriteTilesZarr()
    for region in tiles_list:
        writer.write_tile(tmp_path, region)

    for src in source_set:
        filename = Path(src[len("file://") :]).parent.stem
        array_name = Path(src[len("file://") :]).name

        # check if zarr prediction exists
        zarr_path = tmp_path / f"{filename}_output.zarr"
        assert zarr_path.exists()

        # load array and compare with original
        g = zarr.open(zarr_path, mode="r")

        # check sharding and chunking
        if shards is not None:
            assert g[array_name].shards == shards
        if chunks is not None:
            assert g[array_name].chunks == chunks

        # pull array
        pred_array = g[array_name][:]
        data_idx = int(array_name.split("_")[-1])
        expected_array = arrays[data_idx]

        # if channels
        if channels is not None:
            expected_array = expected_array[:, channels]

        # zarr file writer does not save singleton dims if not present in original data
        if "C" not in axes:
            expected_array = expected_array.squeeze(axis=1)
        if "S" not in axes:
            expected_array = expected_array.squeeze(axis=0)

        np.testing.assert_allclose(pred_array, expected_array, rtol=1e-5, atol=0)


# TODO update test once array sources is supported
def test_write_from_array(tmp_path):
    """Test that writing from an array source raises an error."""
    arrays = [np.random.rand(32, 32).astype(np.float32) for _ in range(2)]
    image_stacks = load_arrays(
        source=arrays,
        axes="YX",
    )
    patch_extractor = PatchExtractor(image_stacks)

    strategy = TilingStrategy(
        data_shapes=[image_stacks[0].data_shape],
        patch_size=(8, 8),
        overlaps=(4, 4),
    )

    # use writer to write predictionsz
    writer = WriteTilesZarr()
    with pytest.raises(NotImplementedError):
        for region in gen_image_regions("YX", patch_extractor, strategy):
            writer.write_tile(tmp_path, region)


# TODO update test once tiff sources is supported
def test_write_from_tiff(tmp_path):
    """Test that writing from a tiff source raises an error."""
    # save tiffs
    arrays = [np.random.rand(32, 32).astype(np.float32) for _ in range(2)]
    files = [tmp_path / f"test_{i}.tiff" for i in range(2)]
    for file, array in zip(files, arrays, strict=True):
        tifffile.imwrite(file, array)

    image_stacks = load_tiffs(
        source=files,
        axes="YX",
    )
    patch_extractor = PatchExtractor(image_stacks)

    strategy = TilingStrategy(
        data_shapes=[image_stacks[0].data_shape],
        patch_size=(8, 8),
        overlaps=(4, 4),
    )

    # use writer to write predictions
    writer = WriteTilesZarr()
    with pytest.raises(NotImplementedError):
        for region in gen_image_regions("YX", patch_extractor, strategy):
            writer.write_tile(tmp_path, region)
