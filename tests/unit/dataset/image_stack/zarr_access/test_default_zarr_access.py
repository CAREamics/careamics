import itertools
from contextlib import nullcontext as does_not_raise

import pytest
import zarr
from zarrs import ZarrsCodecPipeline

from careamics.dataset.image_stack.zarr_access import (
    DefaultZarrAccess,
)

from .test_zarr_access_backends import ARRAYS, GROUP_W_ARRAYS


class TestDefaultZarrAccess:
    @pytest.mark.parametrize(
        "array_shape, patch_index, expected",
        [
            pytest.param(
                (8, 2, 128, 129),
                (0, slice(None), slice(3, 67), slice(5, 69)),
                (0, [0, 1], slice(3, 67), slice(5, 69)),
                id="SCYX",
            ),
            pytest.param(
                (8, 128, 129, 2),
                (0, slice(3, 67), slice(5, 69), slice(None)),
                (0, slice(3, 67), slice(5, 69), [0, 1]),
                id="SYXC",
            ),
            pytest.param(
                (2, 128, 129, 8),
                (slice(None), slice(3, 67), slice(5, 69), 0),
                ([0, 1], slice(3, 67), slice(5, 69), 0),
                id="CYXS",
            ),
            pytest.param(
                (8, 2, 4, 128, 129),
                (0, slice(None), slice(1, 3), slice(3, 67), slice(5, 69)),
                (0, [0, 1], slice(1, 3), slice(3, 67), slice(5, 69)),
                id="SCZYX",
            ),
        ],
    )
    def test_normalize_zarrs_patch_index(self, array_shape, patch_index, expected):
        normalized = DefaultZarrAccess._normalize_zarrs_patch_index(
            array_shape, patch_index
        )

        assert normalized == expected

    def test_zarrs_config_is_scoped(self):
        access = DefaultZarrAccess(use_zarrs=True)
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
        access = DefaultZarrAccess()
        node = zarr_nodes[key]
        with expected:
            access._require_array(node)
