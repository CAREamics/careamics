"""OME-Zarr writing helpers shared across Zarr backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import zarr
from yaozarrs import v05

from .ome_zarr_utils import default_ome_axes_metadata
from .zarr_access_protocol import ZarrNode
from .zarr_access_utils import file_uri_to_path

# TODO in the near future we need to decouple OME specs generation and Zarr group/array
# creation.


@dataclass(frozen=True)
class OMEWriteTarget:
    """Resolved OME-Zarr output target."""

    store_uri: str
    """URI pointing to the root of a Zarr store."""

    image_group_path: str
    """Path to the group containing the image. If blank, the root group is used."""

    array_name: str
    """Name of the array within the image group."""

    @property
    def array_node(self) -> ZarrNode:
        """Return the array node for this target.

        Returns
        -------
        ZarrNode
            Array node.
        """
        if self.image_group_path == "":
            path = self.array_name
        else:
            path = f"{self.image_group_path}/{self.array_name}"

        return ZarrNode(
            store_uri=self.store_uri,
            path=path,
            node_type="array",
        )


def resolve_ome_output_node(
    source_node: ZarrNode,
    output_store_uri: str,
    source_ome: dict[str, Any] | None,
) -> ZarrNode:
    """Resolve the output array node for a Zarr source.

    Parameters
    ----------
    source_node : ZarrNode
        Source Zarr node.
    output_store_uri : str
        Output store URI.
    source_ome : dict[str, Any] or None
        Source OME metadata.

    Returns
    -------
    ZarrNode
        Output array node.
    """
    if source_ome is not None:
        image_group_path = source_ome.get("image_group_path", "")
        array_name = source_ome.get("level", "0")
        path = (
            array_name if image_group_path == "" else f"{image_group_path}/{array_name}"
        )
    elif source_node.path == "":
        path = "0"
    else:
        path = f"{source_node.path}/0"

    return ZarrNode(
        store_uri=output_store_uri,
        path=path,
        node_type="array",
    )


def get_ome_dimension_names(
    axes: str,
    source_ome: dict[str, Any] | None,
) -> list[str]:
    """Return output dimension names.

    Parameters
    ----------
    axes : str
        Output axes.
    source_ome : dict[str, Any] or None
        Source OME metadata.

    Returns
    -------
    list[str]
        Output dimension names.
    """
    if source_ome is not None:
        axis_metadata = source_ome.get("axes")
        if isinstance(axis_metadata, list) and len(axis_metadata) == len(axes):
            dimension_names = [axis.get("name") for axis in axis_metadata]
            if all(isinstance(name, str) for name in dimension_names):
                return dimension_names

    return [axis_name.lower() for axis_name in axes]


def to_ome_write_target(node: ZarrNode) -> OMEWriteTarget:
    """Resolve an OME-Zarr write target from an array node.

    Parameters
    ----------
    node : ZarrNode
        Destination array node.

    Returns
    -------
    OMEWriteTarget
        OME output target.
    """
    if node.path == "":
        return OMEWriteTarget(
            store_uri=node.store_uri,
            image_group_path="",
            array_name="0",
        )

    image_group_path, _, array_name = node.path.rpartition("/")
    return OMEWriteTarget(
        store_uri=node.store_uri,
        image_group_path=image_group_path,
        array_name=array_name,
    )


def _ensure_group(root: zarr.Group, group_path: str) -> zarr.Group:
    """Ensure a nested group exists.

    Parameters
    ----------
    root : zarr.Group
        Root group.
    group_path : str
        Nested group path.

    Returns
    -------
    zarr.Group
        Existing or newly created group.
    """
    if group_path == "":
        return root

    group = root
    for part in group_path.split("/"):
        if part not in group:
            group = group.create_group(part)
        else:
            child = group[part]
            if not isinstance(child, zarr.Group):
                raise RuntimeError(f"Zarr node '{group_path}' is not a group.")
            group = child

    return group


def _build_image_metadata(
    axes: str,
    source_ome: dict[str, Any] | None,
    array_name: str,
) -> dict[str, Any]:
    """Build OME image metadata for one output image group.

    Parameters
    ----------
    axes : str
        CAREamics target axes.
    source_ome : dict[str, Any] or None
        Source OME metadata, if any.
    array_name : str
        Output array name.

    Returns
    -------
    dict[str, Any]
        Group attributes for the image group.
    """
    default_axes = default_ome_axes_metadata(axes)
    if source_ome is not None:
        source_axes = source_ome.get("axes")
        if isinstance(source_axes, list):
            if len(source_axes) != len(axes):
                raise ValueError(
                    "OME axis metadata length does not match prediction axes: "
                    f"got {len(source_axes)} metadata axes for '{axes}'."
                )

        transforms = source_ome.get("coordinate_transformations")
    else:
        # source is not OME-NGFF, use defaults
        source_axes = default_axes
        transforms = [{"type": "scale", "scale": [1.0] * len(source_axes)}]

    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                axes=source_axes,
                datasets=[
                    v05.Dataset(
                        path=array_name,
                        coordinateTransformations=transforms,
                    )
                ],
            )
        ]
    )

    group_json = v05.OMEZarrGroupJSON(attributes=v05.OMEAttributes(ome=image))
    return group_json.model_dump(by_alias=True, exclude_none=True)["attributes"]


def _build_collection_metadata(
    image_group_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build OME collection metadata for a root store and its `OME` group.

    Parameters
    ----------
    image_group_paths : list[str]
        Image-group paths in the collection.

    Returns
    -------
    dict[str, Any]
        Root-group attributes.
    dict[str, Any]
        `OME` group attributes.
    """
    root_json = v05.OMEZarrGroupJSON(
        attributes=v05.OMEAttributes(ome=v05.Bf2Raw(**{"bioformats2raw.layout": 3}))
    )
    series_json = v05.OMEZarrGroupJSON(
        attributes=v05.OMEAttributes(ome=v05.Series(series=image_group_paths))
    )
    return (
        root_json.model_dump(by_alias=True, exclude_none=True)["attributes"],
        series_json.model_dump(by_alias=True, exclude_none=True)["attributes"],
    )


def ensure_ome_store_structure(
    target: OMEWriteTarget,
    axes: str,
    source_ome: dict[str, Any] | None,
    image_group_paths: list[str],
) -> None:
    """Ensure that an output store contains valid OME metadata.

    Parameters
    ----------
    target : OMEWriteTarget
        Output target.
    axes : str
        CAREamics target axes.
    source_ome : dict[str, Any] or None
        Source OME metadata, if any.
    image_group_paths : list[str]
        All image-group paths currently registered for the store.

    Returns
    -------
    None
        Metadata is written in place.
    """
    store_path = file_uri_to_path(target.store_uri)
    root = zarr.open_group(store_path, mode="a")

    if target.image_group_path == "":
        root.attrs.update(
            _build_image_metadata(
                axes=axes,
                source_ome=source_ome,
                array_name=target.array_name,
            )
        )
        return

    root_attrs, ome_group_attrs = _build_collection_metadata(sorted(image_group_paths))
    root.attrs.update(root_attrs)

    ome_group = _ensure_group(root, "OME")
    ome_group.attrs.update(ome_group_attrs)

    image_group = _ensure_group(root, target.image_group_path)
    image_group.attrs.update(
        _build_image_metadata(
            axes=axes,
            source_ome=source_ome,
            array_name=target.array_name,
        )
    )


def create_ome_array(
    target: OMEWriteTarget,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    dtype: Any,
    dimension_names: list[str],
) -> zarr.Array:
    """Create or open an OME-Zarr array.

    Parameters
    ----------
    target : OMEWriteTarget
        Output target.
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
    store_path = file_uri_to_path(target.store_uri)
    root = zarr.open_group(store_path, mode="a")
    group = _ensure_group(root, target.image_group_path)

    if target.array_name in group:
        array = group[target.array_name]
        if not isinstance(array, zarr.Array):
            raise RuntimeError(f"Zarr array '{target.array_name}' is not an array.")
        return array

    array = group.create_array(
        name=target.array_name,
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype=dtype,
        dimension_names=dimension_names,
    )
    if not isinstance(array, zarr.Array):
        raise RuntimeError(f"Zarr array '{target.array_name}' is not an array.")
    return array
