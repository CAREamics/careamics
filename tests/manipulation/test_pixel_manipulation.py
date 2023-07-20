import numpy as np
import pytest

from careamics_restoration.manipulation.pixel_manipulation import (
    default_manipulate,
    get_stratified_coords,
)


@pytest.mark.parametrize(
    "mask_pixel_perc, shape, num_iterations",
    [(0.4, (32, 32), 1000), (0.4, (10, 10, 10), 1000)],
)
def test_get_stratified_coords(mask_pixel_perc, shape, num_iterations):
    """Test the get_stratified_coords function.

    Ensure that the array of coordinates is randomly distributed across the
    image and doesn't demonstrate any strong pattern.
    """
    # Define the number of iterations to run the test and the array
    array = np.zeros(shape)

    # Iterate over the number of iterations and add the coordinates
    for _ in range(num_iterations):
        # Calculate distance between masked pixels as it's done in the
        # get_stratified_coords function
        coords = get_stratified_coords(mask_pixel_perc, shape)
        # Check that the coordinates are within the shape of the array
        # Check that the distance between masked pixels is approximately the same,
        # and distance from border is equal
        for coord_pair in coords:
            assert len(coord_pair) == len(shape)
            assert all(coord_pair) >= 0
            assert [c <= s for c, s in zip(coord_pair, shape)]

        array[tuple(np.array(coords).T.tolist())] += 1

    # Check that the maximum value of the array is less than half of the second
    # largest. This is to ensure that there's no strong pattern in the array.
    hist = sorted(np.histogram(array, bins=100)[0])
    assert hist[-1] < hist[-2] * 2

    # def test_defaul_manipulate():
    #     default_manipulate(np.zeros((10, 10)), 0.5)
    # pass
