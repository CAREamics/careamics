import numpy as np
import pytest

from careamics.dataset_ng.image_stack.image_utils.image_stack_utils import (
    channel_slice,
    pad_patch,
)

data = np.array(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    ]
)


@pytest.mark.parametrize(
    "data_source",
    [data.copy()],
)
@pytest.mark.parametrize(
    "coords, patch_size, expected_patch",
    [
        ((-1, 0), (2, 2), np.array([[[0, 0], [1, 2]]])),
        ((1, 1), (3, 3), np.array([[[5, 6, 0], [8, 9, 0], [0, 0, 0]]])),
        ((0, 0), (3, 3), data.copy()),
        ((-4, 5), (2, 2), np.zeros((1, 2, 2), dtype=data.dtype)),
        (
            (-1, -1),
            (3, 5),
            np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]]),
        ),
    ],
)
def test_pad_patch(data_source, coords, patch_size, expected_patch):
    patch = data_source[
        (
            slice(None, None, None),  # channel axis
            *[
                slice(np.clip(c, 0, s), np.clip(c + ps, 0, s), None)
                for c, ps, s in zip(
                    coords, patch_size, data_source.shape[1:], strict=False
                )
            ],
        )
    ]
    patch_padded = pad_patch(coords, patch_size, (1, *data_source.shape), patch)
    np.testing.assert_equal(patch_padded, expected_patch)


def test_channel_slice():
    """Test channel_slice utility function."""
    shape = (5, 32, 32)
    array = np.arange(np.prod(shape)).reshape(shape)

    # Test single channel
    single_channel = array[channel_slice([3])]
    assert single_channel.shape[0] == 1
    np.testing.assert_array_equal(single_channel, array[3:4])

    # Test multiple channels
    multiple_channels = array[channel_slice([0, 3, 4])]
    assert multiple_channels.shape[0] == 3
    np.testing.assert_array_equal(multiple_channels, array[[0, 3, 4]])

    # Test all channels
    all_channels = array[channel_slice(None)]  # only one ellipsis allowed
    assert all_channels.shape[0] == shape[0]
    np.testing.assert_array_equal(all_channels, array)


def test_error_empty_channel_slice():
    """Test that channel_slice raises an error for empty channel list."""
    with pytest.raises(ValueError):
        _ = channel_slice([])
