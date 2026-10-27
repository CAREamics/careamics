import itertools
from contextlib import nullcontext as does_not_raise

import pytest
import zarr
from zarrs import ZarrsCodecPipeline

from careamics.dataset.image_stack.zarr_access import (
    ZarrPythonAccess,
)

from .test_zarr_access_backends import ARRAYS, GROUP_W_ARRAYS


class TestZarrPythonAccess:
    def test_zarrs_config_is_scoped(self):
        access = ZarrPythonAccess(use_zarrs=True)
        original_pipeline = zarr.config.get("codec_pipeline.path")
        original_strict = zarr.config.get("codec_pipeline.strict", None)

        with access._zarr_config():
            assert zarr.registry.get_pipeline_class() is ZarrsCodecPipeline
            assert zarr.config.get("codec_pipeline.strict") is True

        assert zarr.config.get("codec_pipeline.path") == original_pipeline
        assert zarr.config.get("codec_pipeline.strict", None) == original_strict

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
