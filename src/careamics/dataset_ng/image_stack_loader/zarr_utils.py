import warnings
from pathlib import Path
from urllib.parse import urlparse

import zarr

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

    valid_schemes = {"file", "s3", "gs", "az", "https", "http", "zip"}

    if parsed.scheme and parsed.scheme.lower() in valid_schemes:
        return True

    return False


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
    listof str
        A list of Zarr arrays contained in the group as relative path to the group.
    """
    arrays: list[str] = []

    for name in zarr_group.array_keys():
        if isinstance(zarr_group[name], zarr.Array):
            arrays.append(name)

    if arrays == []:
        warnings.warn(
            f"No arrays found in zarr group at '{zarr_group.path}'.",
            UserWarning,
            stacklevel=2,
        )

    return arrays


def decipher_zarr_uri(source: str) -> tuple[str, str, str]:
    """Extract the zarr store path, group path and array path from a zarr source string.

    The input string is expected to be in the format:
    \"file://path/to/zarr_store.zarr/group/path/array_name\"

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
    key = "file://"

    if source[: len(key)] != key:
        raise ValueError(f"Remote file not supported: {source}")

    if ".zarr" not in source:
        raise ValueError(f"No .zarr file extension found in source: {source}")

    _source = source[len(key) :]
    groups = _source.split("/")

    # find .zarr entry
    zarr_index = next((i for i, p in enumerate(groups) if p.endswith(".zarr")))

    path_to_zarr = groups[: zarr_index + 1]
    parent_path = groups[zarr_index + 1 : -1]
    content_path = groups[-1]

    return "/".join(path_to_zarr), "/".join(parent_path), content_path


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
