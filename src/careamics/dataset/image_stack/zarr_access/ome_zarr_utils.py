"""OME-Zarr helpers shared across Zarr backends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, TypeAlias

import zarr
from pydantic import ValidationError
from yaozarrs import validate_ome_object
from yaozarrs.v05 import Bf2Raw, Dataset, Image, LabelImage, Plate, Series

from .zarr_access_protocol import ZarrNode
from .zarr_access_utils import file_uri_to_path

OMEType: TypeAlias = Image | Bf2Raw | Plate | LabelImage | Series

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


# --- OME-NGFF metadata


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
            f"Expected a single multiscale pyramid, got {len(multiscales)}"
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


def _image_metadata_to_dict(ome_metadata: Image) -> dict[str, Any]:
    """Return a dict from a yaozarrs.Image OME-NGFF metadata.

    Parameters
    ----------
    ome_metadata : Image
        OME-NGFF metadata for an image.

    Returns
    -------
    dict[str, Any]
        OME-NGFF metadata as a dictionary.
    """
    return ome_metadata.model_dump(mode="json", by_alias=True, exclude_none=True)


# ---


@dataclass(frozen=True)
class ResolvedOMEZarrNode:
    """Resolved OME-Zarr array node and metadata."""

    node: ZarrNode
    additional_metadata: dict[str, Any]


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


# TODO revisit
def _build_ome_additional_metadata(
    image_group_path: str,
    dataset_path: str,
    ome_metadata: Image,
    dataset_metadata: Dataset,
) -> dict[str, Any]:
    """Build CAREamics metadata for an OME-resolved array.

    Parameters
    ----------
    image_group_path : str
        Path to the OME image group within the store.
    dataset_path : str
        Selected multiscale dataset path.
    ome_metadata : dict[str, Any]
        OME image metadata.
    dataset_metadata : dict[str, Any]
        Selected dataset metadata.

    Returns
    -------
    dict[str, Any]
        Additional metadata for the resolved image stack.
    """
    multiscales = ome_metadata.multiscales
    axes = multiscales[0].axes

    return {
        "ome": {
            "is_ome_zarr": True,
            "version": ome_metadata.version,
            "image_group_path": image_group_path,
            "multiscale_level": dataset_path,
            "axes": axes,
            "dimension_names": [axis.name for axis in axes],
            "dataset": dataset_metadata,
            "multiscales": multiscales,
        }
    }


def resolve_ome_zarr_nodes(
    node: ZarrNode,
    level: str = "0",
) -> list[ResolvedOMEZarrNode]:
    """Resolve OME-Zarr root or image-group nodes to concrete array nodes.

    Parameters
    ----------
    node : ZarrNode
        Zarr group node to resolve.
    level : str, default="0"
        Multiscale level to select.

    Returns
    -------
    list[ResolvedOMEZarrNode]
        Resolved array nodes with associated OME metadata.
    """
    if node.node_type is not None and node.node_type != "group":
        raise ValueError(
            f"Wrong ZarrNode node type, expected `group`, got `{node.node_type}`."
        )

    group = _open_group(node)
    ome_metadata = _get_ome_metadata(group)
    if ome_metadata is None:
        return []

    # raise error for unsupported Plate and LabelImage
    _raise_for_unsupported_ome_metadata(ome_metadata, node.source)

    # single multiscale Image
    if isinstance(ome_metadata, Image):
        dataset_path, _ = _resolve_dataset(ome_metadata, level)
        image_group_path = node.path
        resolved_path = (
            dataset_path
            if image_group_path == ""
            else f"{image_group_path}/{dataset_path}"
        )
        return [
            ResolvedOMEZarrNode(
                node=ZarrNode(
                    store_uri=node.store_uri,
                    path=resolved_path,
                    node_type="array",
                ),
                additional_metadata=_image_metadata_to_dict(ome_metadata),
            )
        ]
    # collection of images
    elif isinstance(ome_metadata, Bf2Raw):
        resolved_nodes: list[ResolvedOMEZarrNode] = []
        for image_group_path in _get_collection_image_paths(group):
            child_node = ZarrNode(
                store_uri=node.store_uri,
                path=image_group_path,
                node_type="group",
            )
            resolved_nodes.extend(resolve_ome_zarr_nodes(child_node, level=level))
        return resolved_nodes

    return []


def get_ome_array_metadata(node: ZarrNode) -> dict[str, Any]:
    """Return OME metadata for an OME array node.

    Parameters
    ----------
    node : ZarrNode
        Array node to inspect.

    Returns
    -------
    dict[str, Any]
        Additional metadata for the explicit array.
    """
    if node.node_type is not None and node.node_type != "array":
        raise ValueError(
            f"Wrong ZarrNode node type, expected `array`, got `{node.node_type}`."
        )

    # arrays in root are not OME compatible
    if node.path == "":
        return {}

    parent_node = ZarrNode(
        store_uri=node.store_uri,
        path=node.parent_path,
        node_type="group",
    )

    try:
        resolved_nodes = resolve_ome_zarr_nodes(parent_node, level=node.basename)
    except (ValueError, NotImplementedError):
        return {}

    for resolved in resolved_nodes:
        if resolved.node.path == node.path:
            return resolved.additional_metadata

    return {}
