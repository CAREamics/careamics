"""CLI utilities."""

import argparse
from collections.abc import Callable
from typing import Any, Literal

SubcommandFunc = Callable[[argparse.Namespace], None]
Subcommand = Literal["train", "predict", "config-care", "config-n2v", "config-n2n"]


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
