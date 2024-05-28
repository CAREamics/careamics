"""Utility functions for paths."""

from pathlib import Path
from typing import Union


def check_path_exists(path: Union[str, Path]) -> Path:
    """Check if a path exists. If not, raise an error.

    Note that it returns `path` as a Path object.

    Parameters
    ----------
    path : Union[str, Path]
        Path to check.

    Returns
    -------
    Path
        Path as a Path object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data path {path} is incorrect or does not exist.")

    return path
