from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from careamics.dataset_ng.image_stack import ZarrImageStack
from careamics.dataset_ng.image_stack.image_utils import channel_slice

# TODO test _reshaped_data_shape


def create_zarr(
    file_path: Path,
    data_path: str,
    data: NDArray,
    shards: Sequence[int] | None = None,
    chunks: Sequence[int] | None = None,
) -> zarr.Group:
    group = zarr.create_group(file_path.resolve())

    # create array
    array = group.create(
        name=data_path,
        shape=data.shape,
        shards=shards if shards is not None else None,
        chunks=data.shape if chunks is None else chunks,
        dtype=np.uint16,
    )
    # write data
    array[...] = data

    return group


@pytest.mark.parametrize(
    "axes, shape, channels",
    [
        ("CYX", (1, 32, 32), None),
        ("CYX", (1, 32, 32), [0]),
        ("CYX", (5, 32, 32), None),
        ("CYX", (5, 32, 32), [1, 4]),
        ("CYX", (5, 32, 32), [2]),
    ],
)
def test_extract_channels(
    tmp_path: Path,
    axes: str,
    shape: tuple[int, ...],
    channels: int,
):
    # reference data to compare against, it is reshaped using careamics reshape_array
    data = np.arange(np.prod(shape)).reshape(shape)

    # save data as a zarr array to ininitialise image stack with
    file_path = tmp_path / "test_zarr.zarr"
    data_path = "image"

    # initialise ZarrImageStack
    group = create_zarr(file_path=file_path, data_path=data_path, data=data)
    image_stack = ZarrImageStack(group=group, data_path=data_path, axes=axes)

    # extract patch
    patch = image_stack.extract_channel_patch(
        sample_idx=0,
        channels=channels,
        coords=(0, 0),
        patch_size=(8, 8),
    )
    assert len(patch.shape) == 3  # no Z
    assert patch.shape[0] == len(channels) if channels is not None else data.shape[0]

    expected_patch = data[
        channel_slice(channels),
        0 : 0 + 8,
        0 : 0 + 8,
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
    # reference data to compare against, it is reshaped using careamics reshape_array
    data = np.arange(np.prod(shape)).reshape(shape)

    # save data as a zarr array to ininitialise image stack with
    file_path = tmp_path / "test_zarr.zarr"
    data_path = "image"

    # initialise ZarrImageStack
    group = create_zarr(file_path=file_path, data_path=data_path, data=data)
    image_stack = ZarrImageStack(group=group, data_path=data_path, axes=axes)

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


def test_shards_and_chunks(tmp_path: Path):

    axes = "SCZYX"
    shape = (10, 3, 32, 64, 64)
    chunks = (1, 1, 4, 8, 8)
    shards = (1, 1, 8, 32, 16)

    data = np.arange(np.prod(shape)).reshape(shape)

    # save data as a zarr array to ininitialise image stack with
    file_path = tmp_path / "test_zarr.zarr"
    data_path = "image"

    # initialise ZarrImageStack
    group = create_zarr(
        file_path=file_path,
        data_path=data_path,
        data=data,
        shards=shards,
        chunks=chunks,
    )
    image_stack = ZarrImageStack(group=group, data_path=data_path, axes=axes)

    assert image_stack.chunks == chunks
    assert image_stack.shards == shards
