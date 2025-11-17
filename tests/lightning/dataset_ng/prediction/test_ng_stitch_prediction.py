import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy
from careamics.lightning.dataset_ng.prediction.stitch_prediction import (
    sort_tiles_by_data_index,
    sort_tiles_by_sample_index,
    stitch_prediction,
    stitch_single_prediction,
    stitch_single_sample,
)

# TODO will be redundant once tiling_strategy tests will not use the old stitching


@pytest.fixture
def tiles(n_data, shape, axes) -> tuple[NDArray, list[ImageRegionData]]:
    # create data
    array = np.arange(n_data * np.prod(shape)).reshape((n_data, *shape))

    # create tiling strategy
    if "Z" in axes:
        tile_size = (8, 16, 16)
        overlaps = (2, 4, 4)
    else:
        tile_size = (16, 16)
        overlaps = (4, 4)

    if "S" in axes:
        if "C" in axes:
            shape_with_sc = shape
        else:
            shape_with_sc = (shape[0], 1, *shape[1:])
    else:
        if "C" in axes:
            shape_with_sc = (1, *shape)
        else:
            shape_with_sc = (1, 1, *shape)

    tiling_strategy = TilingStrategy(
        data_shapes=[shape_with_sc] * n_data, tile_size=tile_size, overlaps=overlaps
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    patch_extractor = create_array_extractor(
        source=[array[i] for i in range(n_data)], axes=axes
    )

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tile_spec: TileSpecs = tiling_strategy.get_patch_spec(i)
        tile = patch_extractor.extract_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        tiles.append(
            ImageRegionData(
                data=tile,
                source="array",
                dtype=str(tile.dtype),
                data_shape=shape,
                axes=axes,
                region_spec=tile_spec,
            )
        )
    return array, tiles


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1, 3])
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((32, 32), "YX"),
    ],
)
def test_sort_tiles_by_data_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[0]

    sorted_tiles = sort_tiles_by_data_index(tile_list)
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize(
    "n_data, shape, axes",
    [
        (1, (5, 32, 32), "SYX"),
    ],
)
def test_sort_tiles_by_sample_index(tiles):
    array, tile_list = tiles
    n_data = array.shape[1]

    sorted_tiles = sort_tiles_by_sample_index(tile_list)
    assert len(sorted_tiles.keys()) == n_data
    n_tiles_per_data = [len(sorted_tiles[data_idx]) for data_idx in sorted_tiles.keys()]
    assert all(n_tiles_per_data[0] == n_tiles_per_data[i] for i in range(1, n_data))


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((32, 32), "YX"),
        ((3, 32, 32), "CYX"),
        ((8, 32, 32), "ZYX"),
        ((3, 8, 32, 32), "CZYX"),
    ],
)
def test_stitching_single_sample(tiles):
    array, tile_list = tiles
    array = array.squeeze()  # remove sample dim

    stitched_array = stitch_single_sample(tile_list)
    np.testing.assert_array_equal(stitched_array, array)


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [1])
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((32, 32), "YX"),
        ((3, 32, 32), "CYX"),
        ((5, 32, 32), "SYX"),
        ((8, 32, 32), "ZYX"),
        ((3, 8, 32, 32), "CZYX"),
        ((5, 3, 32, 32), "SCYX"),
        ((5, 8, 32, 32), "SZYX"),
        ((5, 3, 8, 32, 32), "SCZYX"),
    ],
)
def test_stitching_single_prediction(tiles):
    array, tile_list = tiles
    array = array.squeeze()  # remove sample dim if S=1

    stitched_array = stitch_single_prediction(tile_list)
    np.testing.assert_array_equal(stitched_array, array)


# parameters are injected automatically in the tiles fixture
@pytest.mark.parametrize("n_data", [4])
@pytest.mark.parametrize(
    "shape, axes",
    [
        ((32, 32), "YX"),
        ((3, 32, 32), "CYX"),
        ((5, 32, 32), "SYX"),
        ((8, 32, 32), "ZYX"),
        ((3, 8, 32, 32), "CZYX"),
        ((5, 3, 32, 32), "SCYX"),
        ((5, 8, 32, 32), "SZYX"),
        ((5, 3, 8, 32, 32), "SCZYX"),
    ],
)
def test_stitching_predicition(tiles):
    array, tile_list = tiles
    array = array.squeeze()  # remove sample dim if S=1

    stitched_arrays = stitch_prediction(tile_list)
    prediction = np.stack(stitched_arrays)

    np.testing.assert_array_equal(prediction, array)
