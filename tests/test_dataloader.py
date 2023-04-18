import pytest 
import tifffile

import numpy as np

from n2v.dataloader import (
    list_input_source_tiff,
    _calculate_stitching_coords,
    extract_patches_sequential
)



def test_list_input_source_tiff(tmp_path):
    """Test listing tiff files"""
    num_files = 4

    # create np arrays
    arrays = []
    for n in range(num_files):
        arr_list = [
            [i for i in range(2+n)],
            [i*i for i in range(2+n)]
        ]
        arrays.append(np.array(arr_list))

    # create tiff files
    for n, arr in enumerate(arrays):
        tifffile.imwrite(tmp_path / f"test_{n}.tif", arr)

    # open tiff files without stating the number of files
    list_of_files = list_input_source_tiff(tmp_path)
    assert len(list_of_files) == num_files
    assert len(set(list_of_files)) == len(list_of_files)

    # open only 2 tiffs
    list_of_files = list_input_source_tiff(tmp_path, num_files=2)
    assert len(list_of_files) == 2
    assert len(set(list_of_files)) == 2


@pytest.mark.parametrize("tile_coords, last_tile_coords, overlap, expected", 
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
                         ])
def test_calculate_stitching_coords(tile_coords, last_tile_coords, overlap, expected):
    """Test calculating stitching coordinates"""
    expected_slices = [
        slice(expected[2*i], expected[2*i+1]) for i in range(len(overlap))
    ]

    # compute stitching coordinates
    result = _calculate_stitching_coords(tile_coords, last_tile_coords, overlap)
    assert result == expected_slices


@pytest.mark.parametrize("arr_shape, patch_size", 
                         [
                            # Wrong number of dimensions 2D
                            ((10, 10), (5, )),
                            ((10, 10), (5, 5, 5)),
                            ((1, 10, 10), (5, )),
                            ((1, 1, 10, 10), (5, )),

                            # Wrong number of dimensions 3D
                            ((10, 10, 10), (5, 5, 5, 5)),
                            ((1, 10, 10, 10), (5, 5)),
                            ((1, 10, 10, 10), (5, 5, 5, 5)),
                            ((1, 1, 10, 10, 10), (5, 5)),
                            ((1, 1, 10, 10, 10), (5, 5, 5, 5)),

                            # Wrong z patch size
                            ((1, 10, 10), (5, 5, 5)),
                            ((10, 10, 10), (10, 5, 5)),

                            # Wrong YX patch sizes
                            ((10, 10), (10, 5)),
                            ((1, 10, 10), (5, 10)),
                        ])
def test_extract_patches_sequential_invalid_arguments(arr_shape, patch_size):
    arr = np.zeros(arr_shape)

    with pytest.raises(ValueError):
        extract_patches_sequential(arr, patch_size)
