from contextlib import nullcontext as does_not_raise

import pytest
import zarr
from yaozarrs import v05

from careamics.dataset.image_stack.zarr_access import ZarrNode, path_to_file_uri
from careamics.dataset.image_stack.zarr_access.ome_zarr_utils import (
    _default_axes_metadata,
    _get_collection_image_paths,
    _get_ome_metadata,
    _raise_for_unsupported_ome_metadata,
    _resolve_dataset,
    get_ome_array_metadata,
    resolve_ome_zarr_nodes,
)

# --- Test utilities


def default_dataset(axes: str) -> list[v05.Dataset]:
    return [
        v05.Dataset(
            path="0",
            coordinateTransformations=[
                v05.ScaleTransformation(scale=[1.0 for _ in axes])
            ],
        )
    ]


IMAGE_METADATA = v05.Image(
    multiscales=[
        v05.Multiscale(
            name="image",
            axes=[
                v05.SpaceAxis(name="y", unit="micrometer"),
                v05.SpaceAxis(name="x", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="0",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[1.0, 1.0])
                    ],
                ),
                v05.Dataset(
                    path="1",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[2.0, 2.0])
                    ],
                ),
            ],
        )
    ]
)

IMAGE_MULT_METADATA: v05.Image = v05.Image(
    multiscales=[
        v05.Multiscale(
            name="image_0",
            axes=[
                v05.SpaceAxis(name="y", unit="micrometer"),
                v05.SpaceAxis(name="x", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="0",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[1.0, 1.0])
                    ],
                )
            ],
        ),
        v05.Multiscale(
            name="image_1",
            axes=[
                v05.SpaceAxis(name="y", unit="micrometer"),
                v05.SpaceAxis(name="x", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="1",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[2.0, 2.0])
                    ],
                )
            ],
        ),
    ]
)

LABEL_METADATA: v05.LabelImage = v05.LabelImage(
    multiscales=[
        v05.Multiscale(
            name="image",
            axes=[
                v05.SpaceAxis(name="y", unit="micrometer"),
                v05.SpaceAxis(name="x", unit="micrometer"),
            ],
            datasets=[
                v05.Dataset(
                    path="0",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[1.0, 1.0])
                    ],
                ),
                v05.Dataset(
                    path="1",
                    coordinateTransformations=[
                        v05.ScaleTransformation(scale=[2.0, 2.0])
                    ],
                ),
            ],
        ),
    ],
    image_label={},
)

BF2LAYOUT: v05.Bf2Raw = v05.Bf2Raw(**{"bioformats2raw.layout": 3})

OME_SERIES: v05.Series = v05.Series(series=["img_0", "img_1"])

PLATE_METADATA: v05.Plate = v05.Plate(
    plate=v05.PlateDef(
        name="plate",
        rows=[v05.Row(name="A")],
        columns=[v05.Column(name="1")],
        wells=[v05.PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
    ),
)

# See fixtures in conftest.py
FIXT_GROUP_TYPE_PAIRS = [
    # image
    ("image", "", v05.Image),
    # collection
    ("collection", "", v05.Bf2Raw),
    ("collection", "OME", v05.Series),
    ("collection", "img_0", v05.Image),
    ("collection", "img_1", v05.Image),
]


# --- Internal OME representation


@pytest.mark.parametrize("axes", ["ZYX", "CYX", "SCZYX"])
def test_default_axes_metadata(axes):
    """Test that default axes are compatible with OME-NGFF."""
    ome_axes = _default_axes_metadata(axes)

    assert len(ome_axes) == len(axes)
    for original, ome in zip(axes, ome_axes, strict=True):
        assert original.lower() == ome["name"]

    # validate axes
    v05.Multiscale(
        name="0",
        axes=ome_axes,
        datasets=default_dataset(axes),
    )


@pytest.mark.parametrize(
    "ome_metadata, exp",
    [
        (IMAGE_METADATA, does_not_raise()),
        (
            IMAGE_MULT_METADATA,
            pytest.raises(NotImplementedError, match="are not supported"),
        ),
        (PLATE_METADATA, pytest.raises(NotImplementedError, match="are not supported")),
        (LABEL_METADATA, pytest.raises(NotImplementedError, match="are not supported")),
        (OME_SERIES, pytest.raises(NotImplementedError, match="cannot be directly")),
    ],
)
def test_raise_for_unsupported_ome_metadata(ome_metadata, exp):
    with exp:
        _raise_for_unsupported_ome_metadata(ome_metadata, source="")


@pytest.mark.parametrize("group_data", FIXT_GROUP_TYPE_PAIRS)
def test_get_ome_metadata(
    single_image_ome_zarr_path,
    image_collection_ome_zarr_path,
    group_data,
):
    ome_type, int_path, ome_class = group_data
    path = (
        single_image_ome_zarr_path
        if ome_type == "image"
        else image_collection_ome_zarr_path
    )

    group = zarr.open(store=path, path=int_path)
    assert isinstance(_get_ome_metadata(group), ome_class)


@pytest.mark.parametrize(
    "ome_metadata, level, exp",
    [
        (IMAGE_METADATA, "0", does_not_raise()),
        (IMAGE_METADATA, "1", does_not_raise()),
        (IMAGE_METADATA, "2", pytest.raises(ValueError, match="not found")),
        (IMAGE_MULT_METADATA, "0", pytest.raises(ValueError, match="single")),
    ],
)
def test_resolve_dataset(ome_metadata, level, exp):
    with exp:
        found_level, _ = _resolve_dataset(ome_metadata, level)
        assert found_level == level


class TestCollectionImagePaths:

    def test_get_collection_image_paths_missing(self, single_image_ome_zarr_path):
        group = zarr.open(single_image_ome_zarr_path)
        with pytest.raises(ValueError, match="is missing"):
            _ = _get_collection_image_paths(group)

    def test_get_collection_image_paths(self, image_collection_ome_zarr_path):
        group = zarr.open(image_collection_ome_zarr_path)
        series = _get_collection_image_paths(group)
        assert set(series) == {"img_0", "img_1"}


class TestResolveOMEZarrNodes:

    def test_single_image(self, single_image_ome_zarr_path):
        zarr_node = ZarrNode(
            store_uri=path_to_file_uri(single_image_ome_zarr_path), path=""
        )
        assert len(resolve_ome_zarr_nodes(zarr_node)) == 1

    def test_collection_image(self, image_collection_ome_zarr_path):
        zarr_node = ZarrNode(
            store_uri=path_to_file_uri(image_collection_ome_zarr_path), path=""
        )
        assert len(resolve_ome_zarr_nodes(zarr_node)) == 2


class TestGetOMEArrayMetadata:

    def test_single_image(self, single_image_ome_zarr_path):
        zarr_node = ZarrNode(
            store_uri=path_to_file_uri(single_image_ome_zarr_path), path="1"
        )
        resolved_dict = get_ome_array_metadata(zarr_node)
        assert "ome" in resolved_dict
        assert resolved_dict["ome"]["layout"] == "image"
        assert resolved_dict["ome"]["image_group_path"] == ""  # group path is the root
        assert resolved_dict["ome"]["level"] == "1"

    def test_collection_image(self, image_collection_ome_zarr_path):
        zarr_node = ZarrNode(
            store_uri=path_to_file_uri(image_collection_ome_zarr_path), path="img_0/1"
        )
        resolved_dict = get_ome_array_metadata(zarr_node)
        assert "ome" in resolved_dict
        assert resolved_dict["ome"]["layout"] == "collection"
        assert resolved_dict["ome"]["image_group_path"] == "img_0"
        assert resolved_dict["ome"]["level"] == "1"
