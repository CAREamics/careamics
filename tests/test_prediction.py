import pytest
from n2v.prediction import extract_patches_predict, calculate_tile_cropping_coords


@pytest.mark.parametrize(
    "tile_coords, last_tile_coords, overlap, expected",
    [
        # (y, x) [...] (start_y, end_y, start_x, end_x)
        # 2D: top left corner
        ((0, 0), (3, 5), (10, 8), (0, -5, 0, -4)),
        # 2D: top right corner
        ((0, 4), (3, 5), (10, 8), (0, -5, 4, None)),
        # 2D: bottom left corner
        ((2, 0), (3, 5), (10, 8), (5, None, 0, -4)),
        # 2D: bottom right corner
        ((2, 4), (3, 5), (10, 8), (5, None, 4, None)),
        # 2D middle
        ((1, 1), (3, 5), (10, 8), (5, -5, 4, -4)),
        # 3D: front (bottom left)
        ((0, 2, 0), (7, 3, 5), (6, 10, 8), (0, -3, 5, None, 0, -4)),
        # 3D: back (bottom left)
        ((6, 2, 0), (7, 3, 5), (6, 10, 8), (3, None, 5, None, 0, -4)),
    ],
)
def test_calculate_stitching_coords(tile_coords, last_tile_coords, overlap, expected):
    """Test calculating stitching coordinates"""
    expected_slices = [
        slice(expected[2 * i], expected[2 * i + 1]) for i in range(len(overlap))
    ]

    # compute stitching coordinates
    result = calculate_tile_cropping_coords(tile_coords, last_tile_coords, overlap)
    assert result == expected_slices
