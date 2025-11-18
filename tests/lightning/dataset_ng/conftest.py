from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

# TODO can all the NG dataset tests be in the same module and share these fixtures?


@pytest.fixture
def zarr_folder(tmp_path) -> Path:
    # Create a temporary folder with some dummy zarr files
    folder_path = tmp_path / "zarr_data"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture
def target_zarr_folder(tmp_path) -> Path:
    # Create a temporary folder with some dummy zarr files
    folder_path = tmp_path / "target_zarr_data"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture
def arrays() -> NDArray:
    return np.arange(3 * 16 * 16).reshape((3, 16, 16))


@pytest.fixture
def zarr_with_arrays(zarr_folder, arrays) -> str:
    """Three arrays in a single zarr group at the root."""
    path = zarr_folder / "zarr_with_arrays.zarr"
    zarr_file = zarr.create_group(path)
    assert path.exists()

    # write arrays to zarr
    for i in range(arrays.shape[0]):
        zarr_file.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))

    return path


@pytest.fixture
def target_zarr_with_arrays(zarr_folder, arrays) -> str:
    """Three arrays in a single zarr group at the root."""
    path = zarr_folder / "target_zarr_with_arrays.zarr"
    zarr_file = zarr.create_group(path)
    assert path.exists()

    # write arrays to zarr
    for i in range(arrays.shape[0]):
        zarr_file.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))

    return path


@pytest.fixture
def zarr_with_groups(zarr_folder, arrays) -> str:
    """Two groups in a single zarr file, each containing arrays."""
    path = zarr_folder / "zarr_with_groups.zarr"
    zarr_file = zarr.create_group(path)
    assert path.exists()

    # write arrays to zarr in different groups
    for i in range(arrays.shape[0]):
        if i % 2 == 0:
            group_name = "groupA"
        else:
            group_name = "groupB"

        if group_name not in zarr_file:
            group = zarr_file.create_group(group_name)
        else:
            group = zarr_file[group_name]

        group.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))

    for i in range(arrays.shape[0]):
        if i % 2 == 0:
            group_name = "target_groupA"
        else:
            group_name = "target_groupB"

        if group_name not in zarr_file:
            group = zarr_file.create_group(group_name)
        else:
            group = zarr_file[group_name]

        group.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))

    return path


@pytest.fixture
def zarr_with_target_and_mask(tmp_path) -> str:
    """Three arrays in a single zarr group at the root."""
    path = tmp_path / "zarr_with_target_and_mask.zarr"
    zarr_file = zarr.open(path, mode="w")
    assert path.exists()

    # arrays
    arrays = np.arange(3 * 16 * 16).reshape((3, 16, 16))
    targets = np.arange(3 * 16 * 16).reshape((3, 16, 16))
    masks = np.ones((3, 16, 16), dtype=bool)
    val = np.ones((16, 16))

    # exclude central frame
    masks[1] = np.zeros((16, 16), dtype=bool)

    # create groups
    input_group = zarr_file.create_group("input")
    target_group = zarr_file.create_group("target")
    mask_group = zarr_file.create_group("mask")
    val_group = zarr_file.create_group("val")

    # write arrays to zarr
    for i in range(arrays.shape[0]):
        input_group.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))
        target_group.create_array(f"array{i}", data=targets[i], chunks=(8, 8))
        mask_group.create_array(f"array{i}", data=masks[i], chunks=(8, 8))
    val_group.create_array("val_array", data=val, chunks=(8, 8))

    return path
