import numpy as np
import pytest

from careamics_restoration.manipulation.pixel_manipulation import (
    default_manipulate,
    get_stratified_coords,
)


# @pytest.mark.parametrize()
# def test_get_stratified_coords():
#     # TODO test no out of range in coords
#     # TODO test randomness (masked pixel distribution doesn't show any pattern)

#     # coordinate_grid += grid_random_increment outputs oor values
#     get_stratified_coords()
#     pass


# def test_defaul_manipulate():
#     default_manipulate(np.zeros((10, 10)), 0.5)
#     pass
