"""Utilities for inspecting Zarr hierarchies."""

from typing import Literal

import zarr

from .zarr_access_protocol import ZarrNode
from .zarr_access_utils import file_uri_to_path


def open_zarr_node(
    node: ZarrNode, mode: Literal["r", "a", "w"] = "r"
) -> zarr.Array | zarr.Group:
    """Open a Zarr node.

    Parameters
    ----------
    node : ZarrNode
        Node to open.
    mode : {"r", "a", "w"}, default="r"
        Open mode.

    Returns
    -------
    zarr.Array or zarr.Group
        Opened node.

    Raises
    ------
    TypeError
        If an internal path is specified but the node is an array.
    """
    store_path = file_uri_to_path(node.store_uri)
    opened = zarr.open(store_path, mode=mode)

    if node.path == "":
        return opened
    if not isinstance(opened, zarr.Group):
        # zarr root can contain an array (#883)
        raise TypeError(
            f"Zarr store at '{store_path}' is an array, cannot access child path "
            f"'{node.path}'."
        )
    return opened[node.path]


def open_zarr_group(node: ZarrNode, mode: Literal["r", "a", "w"] = "r") -> zarr.Group:
    """Open a Zarr node as a group.

    Parameters
    ----------
    node : ZarrNode
        Group node to open.
    mode : {"r", "a", "w"}, default="r"
        Open mode.

    Returns
    -------
    zarr.Group
        Opened group.

    Raises
    ------
    TypeError
        If the node does not point to a group.
    """
    opened = open_zarr_node(node, mode=mode)
    if not isinstance(opened, zarr.Group):
        raise TypeError(f"Node '{node.source}' is not a zarr.Group.")
    return opened


def resolve_node_type(node: ZarrNode) -> ZarrNode:
    """Return a node with its type populated.

    Parameters
    ----------
    node : ZarrNode
        Node to inspect.

    Returns
    -------
    ZarrNode
        Node with its type populated.
    """
    opened = open_zarr_node(node, mode="r")
    if isinstance(opened, zarr.Array):
        node_type: Literal["array", "group"] = "array"
    elif isinstance(opened, zarr.Group):
        node_type = "group"
    else:
        raise ValueError(
            f"Unsupported Zarr node type for '{node.source}': {type(opened)}."
        )

    return ZarrNode(store_uri=node.store_uri, path=node.path, node_type=node_type)


def list_array_paths(node: ZarrNode) -> list[str]:
    """List first-level arrays beneath a group node.

    Parameters
    ----------
    node : ZarrNode
        Group node to inspect.

    Returns
    -------
    list[str]
        Relative array paths.
    """
    opened = open_zarr_group(node, mode="r")
    return list(opened.array_keys())
