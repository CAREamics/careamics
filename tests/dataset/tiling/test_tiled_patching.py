import numpy as np
import pytest

from careamics.config.tile_information import TileInformation
from careamics.dataset.tiling.tiled_patching import (
    _compute_crop_and_stitch_coords_1d,
    extract_tiles,
)


def check_extract_tiles(array: np.ndarray, tile_size, overlaps):
    """Test extracting patches randomly."""
    tile_data_generator = extract_tiles(array, tile_size, overlaps)

    tiles = []
    all_overlap_crop_coords = []
    all_stitch_coords = []

    # Assemble all tiles and their respective coordinates
    for tile_data in tile_data_generator:
        tile = tile_data[0]

        tile_info: TileInformation = tile_data[1]
        overlap_crop_coords = tile_info.overlap_crop_coords
        stitch_coords = tile_info.stitch_coords

        # add data to lists
        tiles.append(tile)
        all_overlap_crop_coords.append(overlap_crop_coords)
        all_stitch_coords.append(stitch_coords)

        # check tile shape, ignore sample dimension
        assert tile.shape[1:] == tile_size
        assert len(overlap_crop_coords) == len(stitch_coords) == len(tile_size)

    # check that each tile has a unique set of coordinates
    assert len(tiles) == len(all_overlap_crop_coords) == len(all_stitch_coords)

    # check that all values are covered by the tiles
    n_max = np.prod(array.shape)  # maximum value in the array
    unique = np.unique(np.array(tiles))  # unique values in the patches
    assert len(unique) >= n_max


@pytest.mark.parametrize(
    "tile_size, overlaps",
    [
        ((4, 4), (2, 2)),
        ((8, 8), (4, 4)),
    ],
)
def test_extract_tiles_2d(array_2D, tile_size, overlaps):
    """Test extracting tiles for prediction in 2D."""
    check_extract_tiles(array_2D, tile_size, overlaps)


@pytest.mark.parametrize(
    "tile_size, overlaps",
    [
        ((4, 4, 4), (2, 2, 2)),
        ((8, 8, 8), (4, 4, 4)),
    ],
)
def test_extract_tiles_3d(array_3D, tile_size, overlaps):
    """Test extracting tiles for prediction in 3D.

    The 3D array is a fixture of shape (1, 8, 16, 16)."""
    check_extract_tiles(array_3D, tile_size, overlaps)


@pytest.mark.parametrize("axis_size", [32, 35, 40])
@pytest.mark.parametrize("patch_size, overlap", [(16, 4), (8, 6), (16, 8), (32, 24)])
def test_compute_crop_and_stitch_coords_1d(axis_size, patch_size, overlap):
    (
        crop_coords,
        stitch_coords,
        overlap_crop_coords,
    ) = _compute_crop_and_stitch_coords_1d(axis_size, patch_size, overlap)

    # check that the number of patches is sufficient to cover the whole axis and that
    # the number of coordinates is
    # the same for all three coordinate groups
    num_patches = np.ceil((axis_size - overlap) / (patch_size - overlap)).astype(int)
    assert (
        len(crop_coords)
        == len(stitch_coords)
        == len(overlap_crop_coords)
        == num_patches
    )
    # check if 0 is the first coordinate, axis_size is last coordinate in all three
    # coordinate groups
    assert all(
        all((group[0][0] == 0, group[-1][1] == axis_size))
        for group in [crop_coords, stitch_coords]
    )
    # check if neighboring stitch coordinates are equal
    assert all(
        stitch_coords[i][1] == stitch_coords[i + 1][0]
        for i in range(len(stitch_coords) - 1)
    )

    # check that the crop coordinates cover the whole axis
    assert (
        np.sum(np.array(crop_coords)[:, 1] - np.array(crop_coords)[:, 0])
        == patch_size * num_patches
    )

    # check that the overlap crop coordinates cover the whole axis
    assert (
        np.sum(
            np.array(overlap_crop_coords)[:, 1] - np.array(overlap_crop_coords)[:, 0]
        )
        == axis_size
    )

    # check that shape of all cropped tiles is equal
    assert np.array_equal(
        np.array(overlap_crop_coords)[:, 1] - np.array(overlap_crop_coords)[:, 0],
        np.array(stitch_coords)[:, 1] - np.array(stitch_coords)[:, 0],
    )
