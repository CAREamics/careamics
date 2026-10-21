"""Shared utilities for Zarr benchmarks."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from careamics.dataset.image_stack.zarr_access import (
    TensorstoreAccess,
    ZarrAccessProtocol,
    ZarrPythonAccess,
)

SEED = 24

BackendFactory = Callable[[], ZarrAccessProtocol]

BACKENDS: dict[str, BackendFactory] = {
    "zarr": ZarrPythonAccess,
    "tensorstore": TensorstoreAccess,
    "zarrs": partial(ZarrPythonAccess, use_zarrs=True),
}


def positive_int(value: str) -> int:
    """Parse a positive integer.

    Parameters
    ----------
    value : str
        Integer value.

    Returns
    -------
    int
        Parsed positive integer.
    """
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive: {value!r}.")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer.

    Parameters
    ----------
    value : str
        Integer value.

    Returns
    -------
    int
        Parsed non-negative integer.
    """
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative: {value!r}.")
    return parsed


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse a comma-separated shape.

    Parameters
    ----------
    value : str
        Comma-separated positive integers.

    Returns
    -------
    tuple[int, ...]
        Parsed shape.
    """
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid shape: {value!r}.") from error

    if len(shape) == 0 or any(size <= 0 for size in shape):
        raise argparse.ArgumentTypeError(
            f"Shape values must be positive integers: {value!r}."
        )
    return shape


def parse_optional_shape(value: str) -> tuple[int, ...] | None:
    """Parse a shape or ``none``.

    Parameters
    ----------
    value : str
        Comma-separated shape or ``none``.

    Returns
    -------
    tuple[int, ...] or None
        Parsed shape.
    """
    if value.lower() == "none":
        return None
    return parse_shape(value)


def shape_label(shape: Sequence[int] | None) -> str:
    """Return a compact shape label.

    Parameters
    ----------
    shape : sequence of int or None
        Shape to format.

    Returns
    -------
    str
        Shape formatted with ``x`` separators.
    """
    if shape is None:
        return "none"
    return "x".join(str(size) for size in shape)


def layout_is_valid(
    array_shape: Sequence[int],
    chunks: Sequence[int],
    shards: Sequence[int] | None,
) -> bool:
    """Return whether a chunk and shard layout is valid.

    Parameters
    ----------
    array_shape : sequence of int
        Array shape.
    chunks : sequence of int
        Chunk shape.
    shards : sequence of int or None
        Shard shape.

    Returns
    -------
    bool
        Whether the layout can be benchmarked.
    """
    if len(chunks) != len(array_shape):
        return False
    if any(chunk > size for chunk, size in zip(chunks, array_shape, strict=True)):
        return False
    if shards is None:
        return True
    if len(shards) != len(array_shape):
        return False
    return all(
        shard >= chunk and shard % chunk == 0
        for chunk, shard in zip(chunks, shards, strict=True)
    )


def write_results(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    output: Path | None,
) -> None:
    """Write benchmark results as CSV.

    Parameters
    ----------
    rows : iterable of mappings
        Result rows.
    fieldnames : sequence of str
        CSV column names.
    output : pathlib.Path or None
        Output path, or ``None`` for standard output.

    Returns
    -------
    None
        Results are written to the selected stream.
    """
    stream: TextIO
    should_close = output is not None
    if output is None:
        stream = sys.stdout
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("w", newline="", encoding="utf-8")

    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if should_close:
            stream.close()
