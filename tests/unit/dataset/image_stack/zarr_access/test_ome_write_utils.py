from careamics.dataset.image_stack.zarr_access import (
    OMEWriteTarget,
    ZarrNode,
    get_ome_dimension_names,
    path_to_file_uri,
    resolve_ome_output_node,
    to_ome_write_target,
)

# --- Test utilities


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


class TestOMEWriteTarget:

    def test_array_node_at_root(self, tmp_path):
        store_uri = path_to_file_uri(tmp_path / "prediction.zarr")
        target = OMEWriteTarget(
            store_uri=store_uri,
            image_group_path="",
            array_name="0",
        )

        assert target.array_node == ZarrNode(
            store_uri=store_uri,
            path="0",
            node_type="array",
        )

    def test_array_node_in_collection(self, tmp_path):
        store_uri = path_to_file_uri(tmp_path / "prediction.zarr")
        target = OMEWriteTarget(
            store_uri=store_uri,
            image_group_path="img_0",
            array_name="0",
        )

        assert target.array_node == ZarrNode(
            store_uri=store_uri,
            path="img_0/0",
            node_type="array",
        )


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


class TestToOMEWriteTarget:

    def test_root_node(self, tmp_path):
        store_uri = path_to_file_uri(tmp_path / "prediction.zarr")
        node = ZarrNode(store_uri=store_uri, path="", node_type="array")

        target = to_ome_write_target(node)

        assert target == OMEWriteTarget(
            store_uri=store_uri,
            image_group_path="",
            array_name="0",
        )

    def test_single_image_level_node(self, tmp_path):
        store_uri = path_to_file_uri(tmp_path / "prediction.zarr")
        node = ZarrNode(store_uri=store_uri, path="1", node_type="array")

        target = to_ome_write_target(node)

        assert target == OMEWriteTarget(
            store_uri=store_uri,
            image_group_path="",
            array_name="1",
        )

    def test_collection_node(self, tmp_path):
        store_uri = path_to_file_uri(tmp_path / "prediction.zarr")
        node = ZarrNode(store_uri=store_uri, path="img_0/0", node_type="array")

        target = to_ome_write_target(node)

        assert target == OMEWriteTarget(
            store_uri=store_uri,
            image_group_path="img_0",
            array_name="0",
        )
