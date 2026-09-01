"""Shared utilities for local Zarr access."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


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
            f"Unsupported Zarr URI scheme '{parsed.scheme}' in {file_uri}."
        )
    return Path(unquote(parsed.path))


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
