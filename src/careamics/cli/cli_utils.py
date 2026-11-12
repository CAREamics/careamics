"""CLI utilities."""

from typing import Any


def without_none(args: dict[str, Any]) -> dict[str, Any]:
    """Remove options that were not provided on the command line.

    Parameters
    ----------
    args : dict[str, Any]
        Key-value arguments.

    Returns
    -------
    dict[str, Any]
        Key-value arguments with `None` value filtered.
    """
    return {key: value for key, value in args.items() if value is not None}
