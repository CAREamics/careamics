import itertools
from contextlib import nullcontext as does_not_raise

import pytest

from careamics.dataset.image_stack.zarr_access import (
    ZarrPythonAccess,
)

from .test_zarr_access_backends import ARRAYS, GROUP_W_ARRAYS


class TestZarrPythonAccess:
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
