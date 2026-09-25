import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import zarr

from careamics.dataset.image_stack.zarr_access import (
    ZarrNode,
    ZarrPythonAccess,
)

# --- Test utilities

BACKENDS = [ZarrPythonAccess]


def to_dict(shape, dtype, chunks, shards) -> dict:
    """Generate a dictionary with array metadata."""
    return {
        "shape": shape,
        "dtype": np.dtype(dtype),
        "chunks": chunks,
        "shards": shards,
    }


def create_array_store(
    store_path,
    *,
    shape=(8, 8),
    dtype=np.float32,
    chunks=(4, 4),
    shards=None,
    data=None,
):
    """Create a root Zarr array for tests."""
    kwargs = {
        "mode": "w",
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
    }
    if shards is not None:
        kwargs["shards"] = shards

    array = zarr.open_array(store_path, **kwargs)
    if data is not None:
        array[...] = data
    return array


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


NODE_KEYS = ["array", "group_1", "group_2", "root_group", "root_array"]

GROUP_W_ARRAYS = [("root_group", 1), ("group_1", 3), ("group_2", 0)]

ARRAYS = [
    ("array", to_dict((16, 16), np.int32, (4, 4), (8, 8))),
    ("array_1_0", to_dict((6, 6), np.int16, (2, 2), None)),
    ("array_1_1", to_dict((8, 8), np.float64, (4, 4), (8, 8))),
    ("array_1_2", to_dict((5, 5), np.float32, (1, 2), None)),
    ("root_array", to_dict((4, 4), "f4", (2, 2), None)),
]

# --- Unit tests


class TestZarrPythonAccess:
    @pytest.mark.parametrize("node_key", NODE_KEYS)
    def test_open_node(self, zarr_nodes, node_key):
        access = ZarrPythonAccess()
        node = zarr_nodes[node_key]
        opened_node = access.open_node(node)

        if node.node_type == "array":
            assert isinstance(opened_node, zarr.Array)
        else:
            assert isinstance(opened_node, zarr.Group)

    @pytest.mark.parametrize("node_key", GROUP_W_ARRAYS)
    def test_list_arrays(self, zarr_nodes, node_key):
        key, n_arrays = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        array_lst = access.list_array_paths(node)
        assert len(array_lst) == n_arrays

    @pytest.mark.parametrize("node_key", ARRAYS)
    def test_get_array_shape(self, zarr_nodes, node_key):
        key, metadata = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        assert access.get_array_shape(node) == metadata["shape"]

    @pytest.mark.parametrize("node_key", ARRAYS)
    def test_get_array_dtype(self, zarr_nodes, node_key):
        key, metadata = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        assert access.get_array_dtype(node) == metadata["dtype"]

    @pytest.mark.parametrize("node_key", ARRAYS)
    def test_get_array_chunks(self, zarr_nodes, node_key):
        key, metadata = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        assert access.get_array_chunks(node) == metadata["chunks"]

    @pytest.mark.parametrize("node_key", ARRAYS)
    def test_get_array_shards(self, zarr_nodes, node_key):
        key, metadata = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        assert access.get_array_shards(node) == metadata["shards"]

    @pytest.mark.parametrize(
        "node_key, expected",
        list(itertools.product(ARRAYS, ["array"]))
        + list(itertools.product(GROUP_W_ARRAYS, ["group"])),
    )
    def test_resolve_node_type(self, zarr_nodes, node_key, expected):
        key, _ = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        assert access.resolve_node_type(node).node_type == expected

    @pytest.mark.parametrize(
        "node_key, expected",
        list(itertools.product(ARRAYS, [does_not_raise()]))
        + list(
            itertools.product(
                GROUP_W_ARRAYS, [pytest.raises(TypeError, match=r"not a zarr\.Array")]
            )
        ),
    )
    def test_require_array(self, zarr_nodes, node_key, expected):
        key, _ = node_key
        access = ZarrPythonAccess()
        node = zarr_nodes[key]
        with expected:
            access._require_array(node)

    @pytest.mark.parametrize(
        "node_key, patch_index, expected",
        [
            ("array", np.s_[2:6, 3:7], np.zeros((4, 4), dtype=np.int32)),
            ("array_1_0", np.s_[1:4, 2:5], np.zeros((3, 3), dtype=np.int16)),
            ("root_array", np.s_[1:3, 0:2], np.zeros((2, 2), dtype=np.float32)),
        ],
    )
    def test_read_array_patch(self, zarr_nodes, node_key, patch_index, expected):
        access = ZarrPythonAccess()
        node = zarr_nodes[node_key]

        patch = access.read_array_patch(node, patch_index)

        assert isinstance(patch, np.ndarray)
        np.testing.assert_array_equal(patch, expected)

    def test_create_array_root_array(self, tmp_path):
        access = ZarrPythonAccess()
        store_path = tmp_path / "root_output.zarr"
        node = ZarrNode(store_uri=store_path.as_uri(), path="", node_type="array")

        created = access.create_array(
            node=node,
            shape=(6, 6),
            chunks=(3, 3),
            shards=None,
            dtype=np.float32,
        )

        assert isinstance(created, zarr.Array)
        assert created.shape == (6, 6)
        assert created.dtype == np.dtype(np.float32)
        assert created.chunks == (3, 3)
        assert created.shards is None

    def test_create_array_group_backed_array(self, tmp_path):
        access = ZarrPythonAccess()
        store_path = tmp_path / "group_output.zarr"
        node = ZarrNode(
            store_uri=store_path.as_uri(),
            path="predictions/array_0",
            node_type="array",
        )

        created = access.create_array(
            node=node,
            shape=(8, 8),
            chunks=(4, 4),
            shards=(8, 8),
            dtype=np.float64,
        )

        assert isinstance(created, zarr.Array)
        assert created.shape == (8, 8)
        assert created.dtype == np.dtype(np.float64)
        assert created.chunks == (4, 4)
        assert created.shards == (8, 8)

        store = zarr.open(store_path, mode="r")
        assert isinstance(store, zarr.Group)
        assert "predictions" in store
        assert "array_0" in store["predictions"]

    def test_create_array_returns_existing_array(self, tmp_path):
        access = ZarrPythonAccess()
        store_path = tmp_path / "existing_root_array.zarr"
        existing = create_array_store(
            store_path,
            shape=(5, 5),
            dtype=np.float32,
            chunks=(2, 2),
        )
        node = ZarrNode(store_uri=store_path.as_uri(), path="", node_type="array")

        created = access.create_array(
            node=node,
            shape=(5, 5),
            chunks=(2, 2),
            shards=None,
            dtype=np.float32,
        )

        assert isinstance(created, zarr.Array)
        assert created.path == existing.path
        assert created.shape == existing.shape

    def test_write_array_tile(self, tmp_path):
        access = ZarrPythonAccess()
        store_path = tmp_path / "write_tile.zarr"
        create_array_store(
            store_path,
            shape=(8, 8),
            dtype=np.float32,
            chunks=(4, 4),
            data=np.zeros((8, 8), dtype=np.float32),
        )
        node = ZarrNode(store_uri=store_path.as_uri(), path="", node_type="array")
        tile = np.ones((3, 2), dtype=np.float32)

        access.write_array_tile(node, np.s_[2:5, 4:6], tile)

        written = zarr.open(store_path, mode="r")
        assert isinstance(written, zarr.Array)
        np.testing.assert_array_equal(written[2:5, 4:6], tile)
        np.testing.assert_array_equal(
            written[:2, :], np.zeros((2, 8), dtype=np.float32)
        )
