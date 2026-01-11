import random
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tifffile import imwrite

from careamics.config.data import NGDataConfig
from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset_ng.dataset import CareamicsDataset, ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_arrays, load_tiffs
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import TilingStrategy
from careamics.lightning.dataset_ng.prediction.stitch_prediction import (
    group_tiles_by_key,
    stitch_prediction,
    stitch_single_prediction,
    stitch_single_sample,
)


@pytest.fixture
def data_config(data_type, axes, shape, channels) -> NGDataConfig:
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
        data_type=data_type,
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


@pytest.fixture
def tiles(
    tmp_path, data_config: NGDataConfig, n_data, shape
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

    if data_config.data_type == "tiff":
        sources = []
        root = tmp_path / "tiff_data"
        root.mkdir(parents=True, exist_ok=True)

        for i in range(n_data):
            file_path = root / f"array_{i}.tiff"
            sources.append(file_path)

            write_array = array[i]

            imwrite(file_path, write_array)
    else:  # array
        sources = [array[i] for i in range(n_data)]

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
    if data_config.data_type == "tiff":
        image_stacks = load_tiffs(source=sources, axes=data_config.axes)
    else:
        image_stacks = load_arrays(source=sources, axes=data_config.axes)
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


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1, 3])
@pytest.mark.parametrize(
    "data_type, shape, axes, channels",
    [
        ("array", (32, 32), "YX", None),
    ],
)
def test_group_tiles_by_data_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[0]

    sorted_tiles = group_tiles_by_key(tile_list, key="data_idx")
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize(
    "n_data, data_type, shape, axes, channels",
    [
        (1, "array", (5, 32, 32), "SYX", None),
    ],
)
def test_group_tiles_by_sample_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[1]

    sorted_tiles = group_tiles_by_key(tile_list, key="sample_idx")
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "data_type, shape, axes, channels",
    [
        ("array", (32, 32), "YX", None),
        ("array", (3, 32, 32), "CYX", None),
        ("array", (3, 32, 32), "CYX", [1]),
        ("array", (3, 32, 32), "CYX", [0, 2]),
        ("array", (8, 32, 32), "ZYX", None),
        ("array", (3, 8, 32, 32), "CZYX", None),
        ("array", (3, 8, 32, 32), "CZYX", [1]),
        ("array", (3, 8, 32, 32), "CZYX", [0, 2]),
    ],
)
def test_stitching_single_sample(tiles, channels):
    array, tile_list = tiles

    # n_data = 1, so remove data dim
    # no sample dim present, so remove it as well
    array = array.squeeze(axis=0).squeeze(axis=0)

    # adjust channels
    if channels is not None:
        array = array[channels]

    stitched_array = stitch_single_sample(tile_list)
    np.testing.assert_allclose(stitched_array, array, rtol=1e-5, atol=0)


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "data_type, shape, axes, channels",
    [
        ("array", (32, 32), "YX", None),
        ("array", (3, 32, 32), "CYX", None),
        ("array", (3, 32, 32), "CYX", [1]),
        ("array", (3, 32, 32), "CYX", [0, 2]),
        ("array", (5, 32, 32), "SYX", None),
        ("array", (8, 32, 32), "ZYX", None),
        ("array", (3, 8, 32, 32), "CZYX", None),
        ("array", (3, 8, 32, 32), "CZYX", [1]),
        ("array", (3, 8, 32, 32), "CZYX", [0, 2]),
        ("array", (5, 3, 32, 32), "SCYX", None),
        ("array", (5, 3, 32, 32), "SCYX", [1]),
        ("array", (5, 3, 32, 32), "SCYX", [0, 2]),
        ("array", (5, 8, 32, 32), "SZYX", None),
        ("array", (5, 3, 8, 32, 32), "SCZYX", None),
        ("array", (5, 3, 8, 32, 32), "SCZYX", [1]),
        ("array", (5, 3, 8, 32, 32), "SCZYX", [0, 2]),
    ],
)
def test_stitching_single_prediction(tiles, channels):
    array, tile_list = tiles

    # n_data = 1, so remove data dim
    array = array.squeeze(axis=0)

    # adjust channels
    if channels is not None:
        array = array[:, channels]

    stitched_array = stitch_single_prediction(tile_list)
    np.testing.assert_allclose(stitched_array, array, rtol=1e-5, atol=0)


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [4])
@pytest.mark.parametrize(
    "data_type, shape, axes, channels",
    [
        ("tiff", (32, 32), "YX", None),
        ("tiff", (3, 32, 32), "CYX", None),
        ("tiff", (3, 32, 32), "CYX", [1]),
        ("tiff", (3, 32, 32), "CYX", [0, 2]),
        ("tiff", (5, 32, 32), "SYX", None),
        ("tiff", (8, 32, 32), "ZYX", None),
        ("tiff", (3, 8, 32, 32), "CZYX", None),
        ("tiff", (3, 8, 32, 32), "CZYX", [1]),
        ("tiff", (3, 8, 32, 32), "CZYX", [0, 2]),
        ("tiff", (5, 3, 32, 32), "SCYX", None),
        ("tiff", (5, 3, 32, 32), "SCYX", [1]),
        ("tiff", (5, 3, 32, 32), "SCYX", [0, 2]),
        ("tiff", (5, 8, 32, 32), "SZYX", None),
        ("tiff", (5, 3, 8, 32, 32), "SCZYX", None),
        ("tiff", (5, 3, 8, 32, 32), "SCZYX", [1]),
        ("tiff", (5, 3, 8, 32, 32), "SCZYX", [0, 2]),
    ],
)
def test_stitching_prediction(tiles, channels):
    array, tile_list = tiles

    stitched_arrays, data_idx = stitch_prediction(tile_list)

    # adjust for channels
    if channels is not None:
        array = array[:, :, channels]

    prediction = np.stack(stitched_arrays)
    np.testing.assert_allclose(prediction, array, rtol=1e-5, atol=0)

    # test data indices
    for i, data_idx_str in enumerate(data_idx):
        assert Path(data_idx_str).name == f"array_{i}.tiff"


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [4])
@pytest.mark.parametrize(
    "data_type, shape, axes, channels",
    [
        ("tiff", (32, 32), "YX", None),
    ],
)
def test_stitching_prediction_ordering(tiles, channels):
    array, tile_list = tiles

    # shuffle tiles to test ordering
    random.shuffle(tile_list)

    stitched_arrays, sources = stitch_prediction(tile_list)

    # check that data indices are in order
    for i, data_idx_str in enumerate(sources):
        assert Path(data_idx_str).name == f"array_{i}.tiff"
        np.testing.assert_allclose(stitched_arrays[i], array[i], rtol=1e-5, atol=0)
