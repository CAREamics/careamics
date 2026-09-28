"""Shared utilities for local Zarr access."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from zarr.storage import StorePath

from .zarr_access_protocol import ZarrNode


def _is_windows_drive(value: str) -> bool:
    """Return whether a string starts with a Windows drive prefix.

    Parameters
    ----------
    value : str
        URI authority.

    Returns
    -------
    bool
        Whether the URI authority starts with a Windows drive prefix.
    """
    return re.match(r"^[A-Za-z]:", value) is not None


def is_valid_uri(path: str | Path) -> bool:
    """Check if a path is a supported URI.

    Parameters
    ----------
    path : str | Path
        Path or URI to check.

    Returns
    -------
    bool
        True if the input is a supported URI.
    """
    try:
        file_uri_to_path(str(path))
    except ValueError:
        return False
    return True


# TODO from py313, we can use `Path.from_uri(file_uri)`
def file_uri_to_path(file_uri: str) -> Path:
    """Convert a local `file://` URI to a path.

    Parameters
    ----------
    file_uri : str
        Local file URI pointing to a Zarr store.

    Returns
    -------
    Path
        Local filesystem path.
    """
    parsed = urlparse(file_uri)

    if parsed.scheme != "file":
        raise ValueError(
            f"Unsupported Zarr URI scheme {parsed.scheme!r} in {file_uri!r}."
        )

    # a local file URI must have no authority, or use "localhost".
    # on Windows, we can encounter URIs with `file://C:\path`.
    if parsed.netloc in ("", "localhost"):
        path = unquote(parsed.path)
    elif os.name == "nt" and _is_windows_drive(parsed.netloc):
        path = f"{parsed.netloc}{unquote(parsed.path)}"
    elif os.name == "nt":
        # convert file://server/share/path to a Windows UNC path
        path = f"//{parsed.netloc}{unquote(parsed.path)}"
    else:
        raise ValueError(
            f"Non-local file URI authority {parsed.netloc!r} in {file_uri!r}."
        )

    # convert /C:/path to C:/path on Windows
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]

    result = Path(path)

    if not result.is_absolute():
        raise ValueError(f"File URI does not contain an absolute path: {file_uri!r}.")

    return result


def path_to_file_uri(path: str | Path) -> str:
    """Convert a local path to a `file://` URI.

    Parameters
    ----------
    path : str or Path
        Local filesystem path.

    Returns
    -------
    str
        Local file URI.
    """
    return Path(path).resolve().as_uri()


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
    normalized_source_str = source_str.replace("\\", "/")

    if not is_valid_uri(source_str):
        source_path = Path(source_str)
        if source_path.suffix != ".zarr":
            raise ValueError(
                f"Source '{source}' is neither a zarr path nor a file URI."
            )
        return ZarrNode(store_uri=path_to_file_uri(source_path))

    store_suffix = ".zarr"
    zarr_index = normalized_source_str.find(store_suffix)
    if zarr_index == -1:
        raise ValueError(f"No .zarr file extension found in source: {source}")

    store_uri = source_str[: zarr_index + len(store_suffix)]
    node_path = normalized_source_str[zarr_index + len(store_suffix) :].lstrip("/")
    return ZarrNode(store_uri=store_uri, path=node_path)
