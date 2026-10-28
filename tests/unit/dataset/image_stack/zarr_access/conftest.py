from pathlib import Path

import numpy as np
import pytest
import zarr

from careamics.dataset.image_stack.zarr_access import ZarrNode
from tests.conftest import (
    create_image_collection_ome_zarr,
    create_single_image_ome_zarr,
)


@pytest.fixture(scope="session")
def single_image_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_single")
    return create_single_image_ome_zarr(root / "single_image.zarr")


@pytest.fixture(scope="session")
def image_collection_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_collection")
    return create_image_collection_ome_zarr(root / "collection.zarr")


@pytest.fixture(scope="session")
def zarr_nodes(tmp_path_factory):
    root_dir = tmp_path_factory.mktemp("zarr_access")
    group_store = root_dir / "group_store.zarr"
    root_array_store = root_dir / "root_array_store.zarr"

    group = zarr.create_group(group_store)
    group.create_array(
        "array",
        data=np.zeros((16, 16)).astype(np.int32),
        chunks=(4, 4),
        shards=(8, 8),
    )

    group_1 = group.create_group("group_1")
    group_1.create_array(
        "array_1_0",
        data=np.zeros((6, 6)).astype(np.int16),
        chunks=(2, 2),
    )
    group_1.create_array(
        "array_1_1",
        data=np.zeros((8, 8)).astype(np.float64),
        chunks=(4, 4),
        shards=(8, 8),
    )
    group_1.create_array(
        "array_1_2",
        data=np.zeros((5, 5)).astype(np.float32),
        chunks=(1, 2),
    )

    group.create_group("group_1/group_2")

    zarr.open_array(root_array_store, mode="w", shape=(4, 4), chunks=(2, 2), dtype="f4")

    return {
        "array": ZarrNode(
            store_uri=group_store.as_uri(), path="array", node_type="array"
        ),
        "group_1": ZarrNode(
            store_uri=group_store.as_uri(), path="group_1", node_type="group"
        ),
        "array_1_0": ZarrNode(
            store_uri=group_store.as_uri(), path="group_1/array_1_0", node_type="array"
        ),
        "array_1_1": ZarrNode(
            store_uri=group_store.as_uri(), path="group_1/array_1_1", node_type="array"
        ),
        "array_1_2": ZarrNode(
            store_uri=group_store.as_uri(), path="group_1/array_1_2", node_type="array"
        ),
        "group_2": ZarrNode(
            store_uri=group_store.as_uri(), path="group_1/group_2", node_type="group"
        ),
        "root_group": ZarrNode(
            store_uri=group_store.as_uri(), path="", node_type="group"
        ),
        "root_array": ZarrNode(
            store_uri=root_array_store.as_uri(), path="", node_type="array"
        ),
    }
