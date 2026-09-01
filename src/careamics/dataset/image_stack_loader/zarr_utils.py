"""Zarr path and URI utilities for image stack loaders."""

from pathlib import Path
from urllib.parse import urlparse

import zarr
from zarr.storage import StorePath

from careamics.dataset.image_stack.zarr_access import (
    ZarrNode,
    file_uri_to_path,
    path_to_file_uri,
)
from careamics.utils import get_logger

logger = get_logger("ZarrUtils")

INPUT = str | Path


def is_valid_uri(path: str | Path) -> bool:
    """
    Check if a path is a valid URI.

    Parameters
    ----------
    path : str | Path
        The path to check.

    Returns
    -------
    bool
        True if the path is valid URI, False otherwise.
    """
    parsed = urlparse(str(path))

    # TODO: expand once remote Zarr backends are implemented.
    valid_schemes = {"file"}

    if parsed.scheme and parsed.scheme.lower() in valid_schemes:
        return True

    return False


def to_zarr_node(source: str | Path | StorePath) -> ZarrNode:
    """Convert a local path or `file://` URI to a :class:`ZarrNode`.

    Parameters
    ----------
    source : str, Path, or StorePath
        Local path or local file URI.

    Returns
    -------
    ZarrNode
        Parsed Zarr node reference.
    """
    if isinstance(source, Path):
        return ZarrNode(store_uri=path_to_file_uri(source))

    source_str = str(source)

    if not is_valid_uri(source_str):
        source_path = Path(source_str)
        if source_path.suffix != ".zarr":
            raise ValueError(
                f"Source '{source}' is neither a zarr path nor a file URI."
            )
        return ZarrNode(store_uri=path_to_file_uri(source_path))

    parsed = urlparse(source_str)
    store_suffix = ".zarr"
    full_path = parsed.path
    zarr_index = full_path.find(store_suffix)
    if zarr_index == -1:
        raise ValueError(f"No .zarr file extension found in source: {source}")

    store_path = full_path[: zarr_index + len(store_suffix)]
    node_path = full_path[zarr_index + len(store_suffix) :].lstrip("/")
    return ZarrNode(store_uri=path_to_file_uri(store_path), path=node_path)


def collect_arrays(zarr_group: zarr.Group) -> list[str]:
    """
    Collect all arrays in a Zarr group into a list.

    Only run on the first level of the group.

    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group to collect arrays from.

    Returns
    -------
    list[str]
        A list of Zarr arrays contained in the group as relative path to the group.
    """
    arrays: list[str] = []

    for name in zarr_group.array_keys():
        if isinstance(zarr_group[name], zarr.Array):
            arrays.append(name)

    if arrays == []:
        logger.warning(f"No arrays found in zarr group at '{zarr_group.path}'.")

    return arrays


# TODO refactor in dataset.image_stack.zarr_access?
def decipher_zarr_uri(source: str) -> tuple[str, str, str]:
    """Extract the zarr store path, group path and array path from a zarr source.

    The input string is expected to be in the format:
    `file://path/to/zarr_store.zarr/group/path/array_name`.

    Note that the root folder of the zarr store must end with ".zarr".

    Parameters
    ----------
    source : str
        The zarr source string.

    Returns
    -------
    str
        The path to the zarr store.
    str
        The parent group within the zarr store, if it is not the root, else "".
    str
        The group or array name the source is pointing to.

    Raises
    ------
    ValueError
        If the source string does not start with "file://".
    ValueError
        If the source string does not contain a ".zarr" file extension.
    """
    if not is_valid_uri(source):
        raise ValueError(f"Remote file not supported: {source}")

    node = to_zarr_node(source)
    path_to_zarr = str(file_uri_to_path(node.store_uri))
    parent_path = node.parent_path
    content_path = node.basename
    return path_to_zarr, parent_path, content_path


# TODO use yaozarrs models to validate OME-Zarr structure
def is_ome_zarr(zarr_group: zarr.Group) -> bool:
    """Check if a Zarr group is an OME-Zarr.

    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group to check.

    Returns
    -------
    bool
        True if the Zarr group is an OME-Zarr, False otherwise.
    """
    return False
