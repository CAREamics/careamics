import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.image_stack import InMemoryImageStack
from careamics.dataset_ng.image_stack.image_utils import channel_slice


@pytest.fixture
def array_stack(shape, axes) -> tuple[NDArray, InMemoryImageStack]:
    data = np.arange(np.prod(shape)).reshape(shape)
    return data, InMemoryImageStack.from_array(data, axes)


# TODO test reshaping of array


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((2, 3, 64, 64), "SCYX"),
        ((3, 4, 32, 32, 32), "SCZYX"),
    ],
)
def test_from_array(array_stack):
    data, image_stack = array_stack
    np.testing.assert_array_equal(image_stack._data, data)
    assert image_stack.data_shape == data.shape
    assert image_stack.data_dtype == data.dtype


@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((2, 1, 64, 64), "SCYX", None),
        ((2, 1, 64, 64), "SCYX", [0]),
        ((2, 3, 64, 64), "SCYX", None),
        ((2, 3, 64, 64), "SCYX", [0, 2]),
        ((2, 3, 64, 64), "SCYX", [1]),
    ],
)
def test_extract_channels(array_stack, channels):
    data, image_stack = array_stack
    data_shape = data.shape

    patch = image_stack.extract_channel_patch(
        sample_idx=1,
        channels=channels,
        coords=(10, 10),
        patch_size=(32, 32),
    )
    assert len(patch.shape) == 3  # no Z
    assert patch.shape[0] == len(channels) if channels is not None else data_shape[1]

    expected_patch = data[
        1,
        channel_slice(channels),
        10 : 10 + 32,
        10 : 10 + 32,
    ]
    np.testing.assert_array_equal(patch, expected_patch)


@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((2, 3, 64, 64), "SCYX", [0, 4]),
        ((2, 3, 64, 64), "SCYX", [3]),
    ],
)
def test_extract_channel_error(array_stack, channels):
    data, image_stack = array_stack
    data_shape = data.shape

    expected_msg = (
        f"Channel index {channels[-1]} is out of bounds for data with "
        f"{data_shape[1]} channels. Check the provided `channels` "
        f"parameter in the configuration for erroneous channel "
        f"indices."
    )

    with pytest.raises(ValueError, match=expected_msg):
        image_stack.extract_channel_patch(
            sample_idx=0,
            channels=channels,
            coords=(0, 0),
            patch_size=(16, 16),
        )
