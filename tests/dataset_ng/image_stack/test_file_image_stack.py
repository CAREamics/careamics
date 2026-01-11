from pathlib import Path

import numpy as np
import pytest
import tifffile

from careamics.dataset_ng.image_stack import FileImageStack
from careamics.dataset_ng.image_stack.image_utils import channel_slice


def test_extract_patch(tmp_path: Path):
    data_shape = (64, 47)
    axes = "YX"
    data = np.arange(np.prod(data_shape)).reshape(data_shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)

    # extract patch should raise an error if the image stack is not loaded
    with pytest.raises(ValueError):
        image_stack.extract_patch(sample_idx=0, coords=(4, 8), patch_size=(16, 16))

    # call load & extract patch
    image_stack.load()
    patch = image_stack.extract_patch(sample_idx=0, coords=(4, 8), patch_size=(16, 16))
    # confirm as expected against reference data
    np.testing.assert_array_equal(patch, data[np.newaxis, 4:20, 8:24])


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
def test_extract_channels(tmp_path, shape, axes, channels):
    data = np.arange(np.prod(shape)).reshape(shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)

    # call load & extract patch
    image_stack.load()
    patch = image_stack.extract_channel_patch(
        sample_idx=1,
        channels=channels,
        coords=(10, 10),
        patch_size=(32, 32),
    )
    assert len(patch.shape) == 3  # no Z
    assert patch.shape[0] == len(channels) if channels is not None else data.shape[1]

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
def test_extract_channel_error(
    tmp_path: Path,
    shape: tuple[int, ...],
    axes: str,
    channels: int,
):
    data = np.arange(np.prod(shape)).reshape(shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)

    # call load & extract patch
    image_stack.load()

    expected_msg = (
        f"Channel index {channels[-1]} is out of bounds for data with "
        f"{shape[1]} channels. Check the provided `channels` "
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


def test_load_and_close(tmp_path: Path):
    data_shape = (64, 47)
    axes = "YX"
    data = np.arange(np.prod(data_shape)).reshape(data_shape)
    path = tmp_path / "image.tiff"
    tifffile.imwrite(path, data, metadata={"axes": axes})

    image_stack = FileImageStack.from_tiff(path, axes)
    assert not image_stack.is_loaded

    image_stack.load()
    assert image_stack.is_loaded

    image_stack.close()
    assert not image_stack.is_loaded
