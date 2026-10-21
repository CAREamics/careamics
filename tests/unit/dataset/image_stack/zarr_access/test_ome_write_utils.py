from typing import Any

import numpy as np
import zarr
from yaozarrs import validate_zarr_store

from careamics.dataset.image_stack.zarr_access import (
    ZarrNode,
    ensure_ome_store_structure,
    get_ome_dimension_names,
    path_to_file_uri,
    resolve_ome_output_node,
)
from careamics.dataset.image_stack.zarr_access.ome_write_utils import (
    _ensure_group,
    file_uri_to_path,
)

# --- Test utilities


def _create_ome_array(
    node: ZarrNode,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: Any,
    dimension_names: list[str],
) -> zarr.Array:
    """Create an OME-Zarr array.

    Parameters
    ----------
    node : ZarrNode
        Output array node. Its parent path is the OME image-group path and its
        basename is the array name.
    shape : tuple[int, ...]
        Array shape.
    chunks : tuple[int, ...]
        Chunk shape.
    shards : tuple[int, ...] or None
        Shard shape.
    dtype : Any
        Array dtype.
    dimension_names : list[str]
        Dimension names for the array metadata.

    Returns
    -------
    zarr.Array
        Existing or newly created array.
    """
    image_group_path = node.parent_path
    array_name = node.basename
    store_path = file_uri_to_path(node.store_uri)
    root = zarr.open_group(store_path, mode="a")
    group = _ensure_group(root, image_group_path)

    if array_name in group:
        array = group[array_name]
        if not isinstance(array, zarr.Array):
            raise RuntimeError(f"Zarr array '{node.path}' is not an array.")
        return array

    array = group.create_array(
        name=array_name,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_names=dimension_names,
    )
    if not isinstance(array, zarr.Array):
        raise RuntimeError(f"Zarr array '{array_name}' is not an array.")
    return array


def _source_ome_metadata(
    *,
    image_group_path: str = "",
    level: str = "0",
) -> dict:
    return {
        "layout": "image" if image_group_path == "" else "collection",
        "image_group_path": image_group_path,
        "level": level,
        "axes": [
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "coordinate_transformations": [
            {"type": "scale", "scale": [0.5, 0.5]},
        ],
        "version": "0.5",
    }


# --- Unit tests


class TestResolveOMEOutputNode:

    def test_plain_root_array(self, tmp_path):
        source_node = ZarrNode(
            store_uri=path_to_file_uri(tmp_path / "source.zarr"),
            path="",
            node_type="array",
        )
        output_store_uri = path_to_file_uri(tmp_path / "prediction.zarr")

        node = resolve_ome_output_node(source_node, output_store_uri, None)

        assert node == ZarrNode(
            store_uri=output_store_uri,
            path="0",
            node_type="array",
        )

    def test_plain_nested_array(self, tmp_path):
        source_node = ZarrNode(
            store_uri=path_to_file_uri(tmp_path / "source.zarr"),
            path="sample",
            node_type="array",
        )
        output_store_uri = path_to_file_uri(tmp_path / "prediction.zarr")

        node = resolve_ome_output_node(source_node, output_store_uri, None)

        assert node == ZarrNode(
            store_uri=output_store_uri,
            path="sample/0",
            node_type="array",
        )

    def test_source_single_image_ome(self, tmp_path):
        source_node = ZarrNode(
            store_uri=path_to_file_uri(tmp_path / "source.zarr"),
            path="1",
            node_type="array",
        )
        output_store_uri = path_to_file_uri(tmp_path / "prediction.zarr")

        node = resolve_ome_output_node(
            source_node,
            output_store_uri,
            _source_ome_metadata(level="1"),
        )

        assert node == ZarrNode(
            store_uri=output_store_uri,
            path="1",
            node_type="array",
        )

    def test_source_collection_ome(self, tmp_path):
        source_node = ZarrNode(
            store_uri=path_to_file_uri(tmp_path / "source.zarr"),
            path="img_0/1",
            node_type="array",
        )
        output_store_uri = path_to_file_uri(tmp_path / "prediction.zarr")

        node = resolve_ome_output_node(
            source_node,
            output_store_uri,
            _source_ome_metadata(image_group_path="img_0", level="1"),
        )

        assert node == ZarrNode(
            store_uri=output_store_uri,
            path="img_0/1",
            node_type="array",
        )


class TestDimensionNames:

    def test_from_axes(self):
        assert get_ome_dimension_names("SCZYX", None) == ["s", "c", "z", "y", "x"]

    def test_from_ome_metadata(self):
        source_ome = _source_ome_metadata()
        source_ome["axes"][0]["name"] = "row"
        source_ome["axes"][1]["name"] = "col"

        assert get_ome_dimension_names("YX", source_ome) == ["row", "col"]

    def test_falls_back_when_ome_axes_do_not_match(self):
        source_ome = _source_ome_metadata()

        assert get_ome_dimension_names("CYX", source_ome) == ["c", "y", "x"]

    def test_falls_back_when_ome_axis_names_are_invalid(self):
        source_ome = _source_ome_metadata()
        source_ome["axes"] = [{"name": "y"}, {"type": "space"}]

        assert get_ome_dimension_names("YX", source_ome) == ["y", "x"]


class TestOMEStoreWriting:

    def test_single_image_store_validates(self, tmp_path):
        store_path = tmp_path / "prediction.zarr"
        node = ZarrNode(
            store_uri=path_to_file_uri(store_path),
            path="0",
            node_type="array",
        )

        ensure_ome_store_structure(
            array_node=node,
            axes="YX",
            source_ome=_source_ome_metadata(),
            image_group_paths=[],
        )
        array = _create_ome_array(
            node=node,
            shape=(16, 16),
            chunks=(8, 8),
            shards=None,
            dtype=np.float32,
            dimension_names=["y", "x"],
        )

        assert array.shape == (16, 16)
        validate_zarr_store(store_path)

    def test_collection_store_validates(self, tmp_path):
        store_path = tmp_path / "prediction.zarr"
        store_uri = path_to_file_uri(store_path)

        for image_group_path in ("img_0", "img_1"):
            node = ZarrNode(
                store_uri=store_uri,
                path=f"{image_group_path}/0",
                node_type="array",
            )
            ensure_ome_store_structure(
                array_node=node,
                axes="YX",
                source_ome=_source_ome_metadata(image_group_path=image_group_path),
                image_group_paths=["img_0", "img_1"],
            )
            _create_ome_array(
                node=node,
                shape=(16, 16),
                chunks=(8, 8),
                shards=None,
                dtype=np.float32,
                dimension_names=["y", "x"],
            )

        root = zarr.open_group(store_path, mode="r")
        assert sorted(root["OME"].attrs["ome"]["series"]) == ["img_0", "img_1"]
        validate_zarr_store(store_path)
