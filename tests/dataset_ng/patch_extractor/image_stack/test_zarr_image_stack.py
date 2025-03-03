from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from careamics.dataset.dataset_utils import reshape_array
from careamics.dataset_ng.patch_extractor.image_stack import ZarrImageStack


def create_test_zarr(file_path: Path, data_path: str, data: NDArray):
    store = zarr.storage.FSStore(url=file_path.resolve())
    # create array
    array = zarr.create(
        store=store,
        shape=data.shape,
        chunks=data.shape,  # only 1 chunk
        dtype=np.uint16,
        path=data_path,
    )
    # write data
    array[...] = data
    store.close()


@pytest.mark.parametrize(
    "original_axes, original_shape, expected_shape, sample_idx",
    [
        ("YX", (32, 48), (1, 1, 32, 48), 0),
        ("XYS", (48, 32, 3), (3, 1, 32, 48), 1),
        ("SXYC", (3, 48, 32, 2), (3, 2, 32, 48), 1),
        ("CYXT", (2, 32, 48, 3), (3, 2, 32, 48), 2),
        ("CXYTS", (2, 48, 32, 3, 2), (6, 2, 32, 48), 4),
        ("XCSYT", (48, 1, 2, 32, 3), (6, 1, 32, 48), 5),  # crazy one
    ],
)
def test_extract_patch_2D(
    tmp_path: Path,
    original_axes: str,
    original_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    sample_idx: int,
):
    # reference data to compare against, it is reshaped using careamics reshape_array
    data = np.arange(np.prod(original_shape)).reshape(original_shape)
    data_ref = reshape_array(data, original_axes)

    # save data as a zarr array to ininitialise image stack with
    file_path = tmp_path / "test_zarr.zarr"
    data_path = "image"
    create_test_zarr(file_path=file_path, data_path=data_path, data=data)

    # initialise ZarrImageStack
    store = zarr.storage.FSStore(url=file_path)
    image_stack = ZarrImageStack(store=store, data_path=data_path, axes=original_axes)
    # TODO: this assert can move if _reshaped_data_shape is tested separately
    assert image_stack.data_shape == expected_shape

    # test extracted patch matches patch from reference data
    coords = (11, 4)
    patch_size = (16, 9)

    extracted_patch = image_stack.extract_patch(
        sample_idx=sample_idx, coords=coords, patch_size=patch_size
    )
    patch_ref = data_ref[
        sample_idx,
        :,
        coords[0] : coords[0] + patch_size[0],
        coords[1] : coords[1] + patch_size[1],
    ]
    np.testing.assert_array_equal(extracted_patch, patch_ref)


def test_from_ome_zarr():
    # kinda an integration test
    path = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
    image_stack = ZarrImageStack.from_ome_zarr(path=path)  # initialise image stack
    n_channels = image_stack.data_shape[1]
    patch_size = (100, 64, 25)
    patch = image_stack.extract_patch(
        sample_idx=0, coords=(112, 56, 15), patch_size=patch_size
    )
    assert isinstance(patch, np.ndarray)  # make sure patch is numpy
    assert patch.shape == (n_channels, *patch_size)  # extracted patch has expected size
