"""Zarr-related fixtures"""

from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

# TODO create an OME-Zarr fixture


class ZarrSource(TypedDict):
    zarr_file: str | Path
    array_path: str


@pytest.fixture
def ome_zarr_url() -> str:
    """URL to a public OME-Zarr for testing."""
    return "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"


@pytest.fixture
def arrays() -> NDArray:
    return np.arange(3 * 16 * 16).reshape((3, 16, 16))


@pytest.fixture
def zarr_linear(tmp_path, arrays) -> list[ZarrSource]:
    """Three arrays in a single zarr group at the root."""
    path = tmp_path / "linear.zarr"
    zarr_file = zarr.create_group(str(path.absolute()))

    # write arrays to zarr
    source = []
    for i in range(arrays.shape[0]):
        zarr_file.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))

        source.append(
            {
                "zarr_file": str(path.absolute()),
                "array_path": f"array{i}",
            }
        )

    return source


@pytest.fixture
def zarr_groups(tmp_path, arrays) -> list[ZarrSource]:
    """Two groups in a single zarr file, each containing arrays."""
    path = tmp_path / "groups.zarr"
    zarr_file = zarr.create_group(path)

    # write arrays to zarr in different groups
    source = []
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

        source.append(
            {
                "zarr_file": str(path.absolute()),
                "array_path": f"{group_name}/array{i}",
            }
        )

    return source


@pytest.fixture
def zarr_multiple(tmp_path, arrays) -> list[ZarrSource]:
    """Arrays in the root of two different zarr files."""
    source = []

    # write each array to its own zarr file
    for i in range(arrays.shape[0]):

        if i % 2 == 0:
            path = tmp_path / "even_arrays.zarr"
        else:
            path = tmp_path / "odd_arrays.zarr"

        if not path.exists():
            zarr_file = zarr.create_group(path)
        else:
            zarr_file = zarr.open(path)

        zarr_file.create_array(f"array{i}", data=arrays[i], chunks=(8, 8))
        source.append(
            {
                "zarr_file": str(path.absolute()),
                "array_path": f"array{i}",
            }
        )

    return source
