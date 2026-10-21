import itertools

import pytest

from careamics.dataset.image_stack.zarr_access import (
    list_array_paths,
    resolve_node_type,
)

from .test_zarr_access_backends import ARRAYS, GROUP_W_ARRAYS


class TestZarrHierarchyUtils:
    @pytest.mark.parametrize("node_key", GROUP_W_ARRAYS)
    def test_list_arrays(self, zarr_nodes, node_key):
        key, n_arrays = node_key

        assert len(list_array_paths(zarr_nodes[key])) == n_arrays

    @pytest.mark.parametrize(
        "node_key, expected",
        list(itertools.product(ARRAYS, ["array"]))
        + list(itertools.product(GROUP_W_ARRAYS, ["group"])),
    )
    def test_resolve_node_type(self, zarr_nodes, node_key, expected):
        key, _ = node_key

        assert resolve_node_type(zarr_nodes[key]).node_type == expected
