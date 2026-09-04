"""OME-Zarr helpers shared across Zarr backends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import zarr
from pydantic import ValidationError
from yaozarrs import validate_ome_object
from yaozarrs.v05 import Bf2Raw, Dataset, Image, LabelImage, Plate, Series

from .zarr_access_protocol import ZarrNode
from .zarr_access_utils import file_uri_to_path

OMEType: TypeAlias = Image | Bf2Raw | Plate | LabelImage | Series


# --- Internal OME-NGFF metadata representation


@dataclass(frozen=True)
class ResolvedOMEZarrNode:
    """Resolved OME-Zarr array node and metadata."""

    node: ZarrNode
    additional_metadata: dict[str, Any]


@dataclass(frozen=True)
class OMEZarrMetadata:
    """Internal OME-Zarr metadata representation.

    Attributes
    ----------
    layout : {"image", "collection"}
        OME-Zarr layout category for the source. Image corresponds to a single OME-NGFF
        multiscale image, while collection relates to the bioformats2raw layout for
        multi-image storage.
    image_group_path : str
        Path to the OME image group within the store.
    level : str
        Multiscale level path.
    axes : list[dict[str, Any]]
        Serialized OME axis metadata for the selected image.
    coordinate_transformations : list[dict[str, Any]]
        Serialized coordinate transformations for the selected level.
    version : str
        OME-NGFF version string.
    """

    layout: Literal["image", "collection"]
    image_group_path: str
    level: str
    axes: list[dict[str, Any]]
    coordinate_transformations: list[dict[str, Any]]
    version: str = "0.5"

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a dictionary.

        Returns
        -------
        dict[str, Any]
            Metadata as dictionary.
        """
        metadata = {
            "layout": self.layout,
            "image_group_path": self.image_group_path,
            "level": self.level,
            "axes": self.axes,
            "coordinate_transformations": self.coordinate_transformations,
            "version": self.version,
        }
        return metadata

    def to_additional_metadata(self) -> dict[str, Any]:
        """Return the additional metadata payload.

        Returns
        -------
        dict[str, Any]
            Dictionary holding the metadata with the `ome` key.
        """
        return {"ome": self.to_dict()}


# TODO should we impose here the constraints on axes (len >=2,<=5, unique) in OME-NGFF?
def _default_axes_metadata(axes: str) -> list[dict[str, Any]]:
    """Return default OME axis metadata from CAREamics axes.

    Parameters
    ----------
    axes : str
        CAREamics axes.

    Returns
    -------
    list[dict[str, Any]]
        OME axis metadata.
    """
    metadata = []
    for axis_name in axes:
        if axis_name in {"Z", "Y", "X"}:
            metadata.append({"name": axis_name.lower(), "type": "space"})
        elif axis_name == "T":
            metadata.append({"name": "t", "type": "time"})
        elif axis_name == "C":
            metadata.append({"name": "c", "type": "channel"})
        else:
            metadata.append({"name": axis_name.lower(), "type": "custom"})
    return metadata


def build_default_ome_metadata(node: ZarrNode, axes: str) -> OMEZarrMetadata:
    """Build default OME metadata for a non-OME Zarr array.

    Parameters
    ----------
    node : ZarrNode
        Array node.
    axes : str
        CAREamics axes.

    Returns
    -------
    OMEZarrMetadata
        Default OME metadata.
    """
    is_img_layout = node.path == ""

    axes_metadata = _default_axes_metadata(axes)
    return OMEZarrMetadata(
        layout="image" if is_img_layout else "collection",
        image_group_path="" if is_img_layout else node.path,
        level="0",
        axes=axes_metadata,
        coordinate_transformations=[
            {"type": "scale", "scale": [1.0] * len(axes_metadata)}
        ],
    )


def build_ome_metadata(
    image_group_path: str,
    level: str,
    ome_metadata: Image,
) -> OMEZarrMetadata:
    """Build internal OME metadata from validated OME-Zarr metadata.

    Parameters
    ----------
    image_group_path : str
        Path to the OME image group within the store.
    level : str
        Multiscale level.
    ome_metadata : Image
        OME image metadata.

    Returns
    -------
    OMEZarrMetadata
        Internal OME metadata.
    """
    is_img_layout = image_group_path == ""

    _, dataset_metadata = _resolve_dataset(ome_metadata, level)
    coordinate_transformations = dataset_metadata.coordinateTransformations
    transforms = [_model_to_dict(transform) for transform in coordinate_transformations]

    return OMEZarrMetadata(
        layout="image" if is_img_layout else "collection",
        image_group_path=image_group_path,
        level=level,
        axes=[_model_to_dict(axis) for axis in ome_metadata.multiscales[0].axes],
        coordinate_transformations=transforms,
        version=ome_metadata.version,
    )


# --- Supported metadata utilities


def _raise_for_unsupported_ome_metadata(ome_metadata: OMEType, source: str) -> None:
    """Raise errors for unsupported OME metadata.

    Parameters
    ----------
    ome_metadata : dict[str, Any]
        OME metadata.
    source : str
        Source string for the error message.

    Raises
    ------
    NotImplementedError
        If the OME metadata are not supported.
    """
    if isinstance(ome_metadata, Plate):
        raise NotImplementedError(
            f"OME-Zarr plate/HCS hierarchies are not supported yet: '{source}'."
        )
    if isinstance(ome_metadata, LabelImage):
        raise NotImplementedError(
            f"OME-Zarr label hierarchies are not supported yet: '{source}'."
        )

    # guard against a node pointing at an "OME" group containing series
    if isinstance(ome_metadata, Series):
        raise NotImplementedError(
            "Series metadata cannot be directly used for loading images."
        )

    if isinstance(ome_metadata, Image):
        if len(ome_metadata.multiscales) > 1:
            raise NotImplementedError(
                f"Multiple multiscale pyramids within one image are not supported, got "
                f"{len(ome_metadata.multiscales)}."
            )


# --- OME-NGFF metadata operations


def _get_ome_metadata(group: zarr.Group) -> OMEType | None:
    """Return validated OME metadata for a group.

    Parameters
    ----------
    group : zarr.Group
        Group to inspect.

    Returns
    -------
    OMEType
        OME metadata if present.
    """
    attributes = group.attrs.asdict()
    ome_metadata = attributes.get("ome")
    if not isinstance(ome_metadata, dict):
        return None

    try:
        ome_class = validate_ome_object(ome_metadata)
    except ValidationError as error:
        warnings.warn(
            "Malformed OME-NGFF metadata detected in "
            f"'{group.store_path}/{group.path}'. Falling back to generic Zarr loading. "
            f"Validation error: {error}",
            stacklevel=2,
        )
        return None

    return ome_class


def _resolve_dataset(
    ome_metadata: Image,
    level: str,
) -> tuple[str, Dataset]:
    """Resolve a multiscale level to a dataset path.

    Parameters
    ----------
    ome_metadata : yaozarrs.Image
        OME image metadata.
    level : str
        Requested multiscale level.

    Returns
    -------
    str
        Dataset path (level).
    yaozarrs.Dataset
        Dataset metadata.
    """
    multiscales = ome_metadata.multiscales
    if len(multiscales) > 1:
        raise ValueError(
            f"Expected a single multiscale pyramid, got {len(multiscales)}."
        )

    datasets = multiscales[0].datasets
    for dataset in datasets:
        if dataset.path == level:
            return level, dataset

    available_levels = [d.path for d in datasets]
    raise ValueError(
        f"OME-Zarr level '{level}' not found in image metadata, available levels are "
        f"{available_levels}."
    )


def _get_collection_image_paths(group: zarr.Group) -> list[str]:
    """Return image-group paths for an OME image collection.

    Parameters
    ----------
    group : zarr.Group
        Collection root group.

    Returns
    -------
    list[str]
        Image-group paths within the collection.
    """
    if "OME" in group and isinstance(group["OME"], zarr.Group):
        ome_group = group["OME"]
        ome_metadata = _get_ome_metadata(ome_group)
        if ome_metadata is None:
            raise ValueError(
                f"OME-Zarr collection '{group.store_path}' is missing valid "
                "OME/series metadata."
            )
        if not isinstance(ome_metadata, Series):
            raise ValueError(
                f"OME-Zarr collection '{group.store_path}' has invalid OME metadata in "
                "its OME group."
            )
        return ome_metadata.series

    raise ValueError(
        f"OME-Zarr collection '{group.store_path}' is missing its OME group."
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Return a Pydantic model as a dictionary.

    Parameters
    ----------
    model : Any
        Pydantic model.

    Returns
    -------
    dict[str, Any]
        Model as a dictionary.
    """
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


# ---


def _open_group(node: ZarrNode) -> zarr.Group:
    """Open a group ZarrNode.

    Parameters
    ----------
    node : ZarrNode
        Group node to open.

    Returns
    -------
    zarr.Group
        Opened group.

    Raises
    ------
    ValueError
        If the ZarrNode does not represent a Zarr group.
    """
    store_path = file_uri_to_path(node.store_uri)
    opened = zarr.open(store_path, mode="r")

    if node.path == "":
        if not isinstance(opened, zarr.Group):
            raise ValueError(f"Node '{node.source}' is not a zarr.Group.")
        return opened

    if not isinstance(opened, zarr.Group):
        raise ValueError(f"Node '{node.source}' is not a zarr.Group.")

    group = opened[node.path]
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Node '{node.source}' is not a zarr.Group.")

    return group


def resolve_ome_zarr_nodes(
    group_node: ZarrNode,
    level: str = "0",
) -> list[ResolvedOMEZarrNode]:
    """Resolve a Zarr node to concrete OME zarr nodes.

    Since only single multiscale pyramid are supported, this method returns a
    single node for a zarr node containing an array with OME-Zarr Metadata. For
    bioformats2raw.layout, it returns all arrays in the serie as Zarr nodes.

    Parameters
    ----------
    group_node : ZarrNode
        Zarr group node to resolve.
    level : str, default="0"
        Multiscale level to select.

    Returns
    -------
    list[ResolvedOMEZarrNode]
        Resolved array nodes with associated OME metadata.
    """
    if group_node.node_type is not None and group_node.node_type != "group":
        raise ValueError(
            f"Wrong ZarrNode node type, expected `group`, got `{group_node.node_type}`."
        )

    group = _open_group(group_node)
    ome_metadata = _get_ome_metadata(group)
    if ome_metadata is None:
        return []

    # raise error for unsupported Plate and LabelImage
    _raise_for_unsupported_ome_metadata(ome_metadata, group_node.source)

    # single multiscale Image
    if isinstance(ome_metadata, Image):
        dataset_path, _ = _resolve_dataset(ome_metadata, level)
        image_group_path = group_node.path
        resolved_path = (
            dataset_path
            if image_group_path == ""
            else f"{image_group_path}/{dataset_path}"
        )
        return [
            ResolvedOMEZarrNode(
                node=ZarrNode(
                    store_uri=group_node.store_uri,
                    path=resolved_path,
                    node_type="array",
                ),
                additional_metadata=build_ome_metadata(
                    image_group_path=image_group_path,
                    level=dataset_path,
                    ome_metadata=ome_metadata,
                ).to_additional_metadata(),
            )
        ]
    # collection of images
    elif isinstance(ome_metadata, Bf2Raw):
        resolved_nodes: list[ResolvedOMEZarrNode] = []
        for image_group_path in _get_collection_image_paths(group):
            child_node = ZarrNode(
                store_uri=group_node.store_uri,
                path=image_group_path,
                node_type="group",
            )
            resolved_nodes.extend(resolve_ome_zarr_nodes(child_node, level=level))
        return resolved_nodes

    return []


def get_ome_array_metadata(array_node: ZarrNode) -> dict[str, Any]:
    """Return OME metadata for an OME array node.

    Parameters
    ----------
    array_node : ZarrNode
        Array node to inspect.

    Returns
    -------
    dict[str, Any]
        Additional metadata for the explicit array.
    """
    if array_node.node_type is not None and array_node.node_type != "array":
        raise ValueError(
            f"Wrong ZarrNode node type, expected `array`, got `{array_node.node_type}`."
        )

    # arrays in root are not OME compatible
    if array_node.path == "":
        return {}

    parent_node = ZarrNode(
        store_uri=array_node.store_uri,
        path=array_node.parent_path,
        node_type="group",
    )

    try:
        resolved_nodes = resolve_ome_zarr_nodes(parent_node, level=array_node.basename)
    except (ValueError, NotImplementedError):
        return {}

    for resolved in resolved_nodes:
        if resolved.node.path == array_node.path:
            return resolved.additional_metadata

    return {}
