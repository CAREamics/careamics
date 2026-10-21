import numpy as np
import pytest
import zarr
from yaozarrs import validate_zarr_store

from careamics.dataset.image_stack.zarr_access import (
    TensorstoreAccess,
    ZarrArraySpec,
    ZarrNode,
    ZarrPythonAccess,
    ensure_ome_store_structure,
    path_to_file_uri,
)
from tests.unit.dataset.image_stack.zarr_access.test_ome_write_utils import (
    _source_ome_metadata,
)

BACKENDS = [ZarrPythonAccess, TensorstoreAccess]


@pytest.mark.parametrize("access_cls", BACKENDS)
class TestOMEStoreWriting:
    """Test that OME metadta write utils and backends produce viable OME-NGFF."""

    def test_single_image_store_validates(self, tmp_path, access_cls):
        store_path = tmp_path / "prediction.zarr"
        node = ZarrNode(
            store_uri=path_to_file_uri(store_path),
            path="0",
            node_type="array",
        )

        ensure_ome_store_structure(
            node=node,
            axes="YX",
            source_ome=_source_ome_metadata(),
            image_group_paths=[],
        )
        access_cls().create_array(
            node=node,
            spec=ZarrArraySpec(
                shape=(16, 16),
                chunks=(8, 8),
                shards=None,
                dtype=np.float32,
                dimension_names=("y", "x"),
            ),
        )

        validate_zarr_store(store_path)

    def test_collection_store_validates(self, tmp_path, access_cls):
        store_path = tmp_path / "prediction.zarr"
        store_uri = path_to_file_uri(store_path)

        for image_group_path in ("img_0", "img_1"):
            node = ZarrNode(
                store_uri=store_uri,
                path=f"{image_group_path}/0",
                node_type="array",
            )
            ensure_ome_store_structure(
                node=node,
                axes="YX",
                source_ome=_source_ome_metadata(image_group_path=image_group_path),
                image_group_paths=["img_0", "img_1"],
            )
            access_cls().create_array(
                node=node,
                spec=ZarrArraySpec(
                    shape=(16, 16),
                    chunks=(8, 8),
                    shards=None,
                    dtype=np.float32,
                    dimension_names=("y", "x"),
                ),
            )

        root = zarr.open_group(store_path, mode="r")
        assert sorted(root["OME"].attrs["ome"]["series"]) == ["img_0", "img_1"]
        validate_zarr_store(store_path)
