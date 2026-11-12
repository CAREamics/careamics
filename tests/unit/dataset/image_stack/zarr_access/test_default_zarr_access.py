import itertools
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import zarr
from zarrs import ZarrsCodecPipeline

from careamics.dataset.image_stack.zarr_access import (
    DefaultZarrAccess,
)
from careamics.dataset.image_stack.zarr_access.default_zarr_access import (
    _normalize_zarrs_patch_index,
    _read_zarrs_patch,
)

from .test_zarr_access_backends import ARRAYS, GROUP_W_ARRAYS


@pytest.mark.parametrize(
    "channels",
    [pytest.param(None, id="all-channels"), pytest.param([0, 1], id="channel-list")],
)
def test_read_zarrs_patch(tmp_path, channels):
    """Read patch indices with integer, channel, and spatial selectors."""
    shape = (8, 2, 4, 130, 131)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    store_path = tmp_path / "example.zarr"
    zarr.create_array(
        store_path,
        data=data,
        chunks=(1, 1, 2, 64, 64),
        zarr_format=3,
    )
    patch_index = (
        1,
        slice(None) if channels is None else channels,
        slice(1, 3),
        slice(3, 68),
        slice(5, 71),
    )

    with zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "codec_pipeline.strict": True,
        }
    ):
        array = zarr.open_array(store_path, mode="r")
        patch = _read_zarrs_patch(array, patch_index)

    np.testing.assert_array_equal(patch, data[patch_index])


@pytest.mark.parametrize(
    "patch_index, expected",
    [
        pytest.param(
            (0, slice(None), slice(3, 67), slice(5, 69)),
            (slice(0, 1), slice(None), slice(3, 67), slice(5, 69)),
            id="SCYX",
        ),
        pytest.param(
            (0, slice(3, 67), slice(5, 69), slice(None)),
            (slice(0, 1), slice(3, 67), slice(5, 69), slice(None)),
            id="SYXC",
        ),
        pytest.param(
            (slice(None), slice(3, 67), slice(5, 69), 0),
            (slice(None), slice(3, 67), slice(5, 69), slice(0, 1)),
            id="CYXS",
        ),
        pytest.param(
            (0, slice(None), slice(1, 3), slice(3, 67), slice(5, 69)),
            (
                slice(0, 1),
                slice(None),
                slice(1, 3),
                slice(3, 67),
                slice(5, 69),
            ),
            id="SCZYX",
        ),
    ],
)
def test_normalize_zarrs_patch_index(patch_index, expected):
    normalized = _normalize_zarrs_patch_index(patch_index)

    assert normalized == expected


def test_zarrs_config_is_scoped():
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
def test_require_array(zarr_nodes, node_key, expected):
    key, _ = node_key
    access = DefaultZarrAccess()
    node = zarr_nodes[key]
    with expected:
        access._require_array(node)
