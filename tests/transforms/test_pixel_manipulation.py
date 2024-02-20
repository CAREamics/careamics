import numpy as np
import pytest

from careamics.transforms.pixel_manipulation import (
    uniform_manipulate,
    _get_stratified_coords,
    _apply_struct_mask,
    median_manipulate
)


# TODO what if ROI size is larger than a spatial dimension (e.g. Z)

@pytest.mark.parametrize(
    "mask_pixel_perc, shape, num_iterations",
    [
        (0.4, (32, 32), 1000), 
        (0.4, (10, 10, 10), 1000)],
)
def test_get_stratified_coords(mask_pixel_perc, shape, num_iterations):
    """Test the get_stratified_coords function.

    Ensure that the array of coordinates is randomly distributed across the
    image and that most pixels get selected.
    """
    # Define the dummy array
    array = np.zeros(shape)

    # Iterate over the number of iterations and add the coordinates. This is an MC
    # simulation to ensure that the coordinates are randomly distributed and not
    # biased towards any particular region.
    for _ in range(num_iterations):
        # Get the coordinates of the pixels to be masked
        coords = _get_stratified_coords(mask_pixel_perc, shape)
        # Check every pair in the array of coordinates
        for coord_pair in coords:
            # Check that the coordinates are of the same shape as the patch
            assert len(coord_pair) == len(shape)
            # Check that the coordinates are positive values
            assert all(coord_pair) >= 0
            # Check that the coordinates are within the shape of the array
            assert [c <= s for c, s in zip(coord_pair, shape)]

        # Add the 1 to the every coordinate location.
        array[tuple(np.array(coords).T.tolist())] += 1

    # Ensure that there's no strong pattern in the array and sufficient number of
    # pixels is masked.
    assert np.sum(array == 0) < np.sum(shape)


# TODO what is this testing?
@pytest.mark.parametrize("mask", [[[0, 1, 1, 1, 1, 1, 0]]])
def test_apply_struct_mask(mask):
    patch = np.zeros((64, 64))
    coords = _get_stratified_coords(0.2, patch.shape)
    patch = _apply_struct_mask(patch, coords, mask)


@pytest.mark.parametrize("shape", 
    [
        (8, 8),
        (3, 8, 8),
        (8, 8, 8),
        (3, 8, 8, 8)
    ]
)
def test_uniform_manipulate(ordered_array, shape):
    """Test the uniform_manipulate function.

    Ensures that the mask corresponds to the manipulated pixels, and that the 
    manipulated pixels have a value taken from a ROI surrounding them.
    """
    # create the array
    patch = ordered_array(shape)

    # manipulate the array
    transform_patch, mask = uniform_manipulate(
        patch, 
        roi_size=5, 
        mask_pixel_percentage=10
    )

    # find pixels that have different values between patch and transformed patch
    diff_coords = np.array(np.where(patch != transform_patch))

    # find non-zero pixels in the mask
    mask_coords = np.array(np.where(mask == 1))

    # check that the transformed pixels correspond to the masked pixels
    assert np.array_equal(diff_coords, mask_coords)

    # for each pixel masked, check that the manipulated pixel value is within the roi
    for i in range(mask_coords.shape[-1]):
        # get coordinates
        coords = mask_coords[..., i]

        # get roi using slice in each dimension
        slices = tuple(
            [
                slice(max(0, coords[i] - 2), min(shape[i], coords[i] + 3)) 
                for i in range(-coords.shape[0]+1, 0) # range -4, -3, -2, -1
            ]
        )
        roi = patch[(...,) + slices] # TODO ellipsis needed bc singleton dim, might need to go away

        # check that the pixel value comes from the actual roi
        assert transform_patch[tuple(coords)] in roi
