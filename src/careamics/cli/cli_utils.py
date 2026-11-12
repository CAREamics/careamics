"""CLI utilities."""

import argparse
from typing import Any, Literal

Subcommand = Literal["train", "predict", "config-care", "config-n2v", "config-n2n"]


def add_subcommand_parser(
    subparsers: argparse._SubParsersAction,
    name: Subcommand,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    """Add a parser for a supported subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers to which the parser is added.
    name : {"train", "predict", "config-care", "config-n2v", "config-n2n"}
        Name of the subcommand.
    **kwargs : Any
        Additional arguments passed to `subparsers.add_parser`.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the subcommand.
    """
    return subparsers.add_parser(name, **kwargs)


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
