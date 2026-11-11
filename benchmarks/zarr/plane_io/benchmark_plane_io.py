"""Benchmark whole XY-plane reads and writes through ``ZarrAccess`` backends."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parents[1]))

from _benchmark_utils import (
    BACKENDS,
    SEED,
    layout_is_valid,
    non_negative_int,
    parse_optional_shape,
    parse_shape,
    positive_int,
    shape_label,
    write_results,
)

from careamics.dataset.image_stack.zarr_access import ZarrArraySpec, ZarrNode

DEFAULT_ARRAY_SHAPE = (2048, 2048)
DEFAULT_CHUNKS = ((64, 64), (128, 128), (256, 256))
DEFAULT_SHARDS = (None, (256, 256), (512, 512))
DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "result" / "plane_io" / "plane_io_results.csv"
)
RESULT_FIELDS = (
    "operation",
    "backend",
    "array_shape",
    "chunks",
    "shards",
    "repeat",
    "seconds",
    "mib_per_second",
    "validated",
)


def create_store(
    path: Path,
    data: np.ndarray,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> ZarrNode:
    """Create a deterministic input store and return its root node."""
    zarr.create_array(
        store=path,
        data=data,
        chunks=chunks,
        shards=shards,
        zarr_format=3,
        overwrite=True,
    )
    return ZarrNode(store_uri=path.as_uri(), node_type="array")


def plane_index(shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Return a full XY index without triggering zarrs' unbounded-slice path."""
    return (slice(0, shape[0]), slice(0, shape[1]))


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the plane read/write benchmark matrix."""
    if len(args.array_shape) != 2:
        raise ValueError("The plane benchmark expects a two-dimensional XY shape.")

    rng = np.random.default_rng(SEED)
    data = rng.integers(0, 256, size=args.array_shape, dtype=np.uint16)
    index = plane_index(args.array_shape)
    expected_plane = data
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="careamics-zarr-plane-") as temp_dir:
        root = Path(temp_dir)
        case_index = 0
        for chunks in args.chunks:
            for shards in args.shards:
                if not layout_is_valid(args.array_shape, chunks, shards):
                    print(
                        f"Skipping incompatible layout chunks={chunks}, "
                        f"shards={shards}.",
                        file=sys.stderr,
                    )
                    continue

                input_path = root / f"input_{case_index}.zarr"
                input_node = create_store(input_path, data, chunks, shards)
                case_index += 1

                for backend_name in args.backends:
                    for operation in args.operations:
                        print(
                            f"Benchmarking {operation} with {backend_name}: "
                            f"chunks={chunks}, shards={shards}.",
                            file=sys.stderr,
                        )
                        access = BACKENDS[backend_name]()
                        if operation == "read":
                            for _ in range(args.warmups):
                                access.read_array_patch(input_node, index)
                        for repeat in range(args.repeats):
                            if operation == "read":
                                start = perf_counter()
                                result = access.read_array_patch(  # type: ignore[attr-defined]
                                    input_node, index
                                )
                                total_bytes = int(result.nbytes)
                                seconds = perf_counter() - start
                                np.testing.assert_array_equal(result, expected_plane)
                            else:
                                output_path = root / (
                                    f"output_{case_index}_{backend_name}_{repeat}.zarr"
                                )
                                output_node = ZarrNode(
                                    store_uri=output_path.as_uri(), node_type="array"
                                )
                                try:
                                    access.create_array(  # type: ignore[attr-defined]
                                        output_node,
                                        ZarrArraySpec(
                                            shape=data.shape,
                                            chunks=chunks,
                                            shards=shards,
                                            dtype=data.dtype,
                                        ),
                                    )
                                    start = perf_counter()
                                    access.write_array_tile(  # type: ignore[attr-defined]
                                        output_node, index, expected_plane
                                    )
                                    seconds = perf_counter() - start
                                    total_bytes = int(expected_plane.nbytes)
                                    result = access.read_array_patch(  # type: ignore[attr-defined]
                                        output_node, index
                                    )
                                    np.testing.assert_array_equal(
                                        result, expected_plane
                                    )
                                finally:
                                    shutil.rmtree(output_path, ignore_errors=True)
                            rows.append(
                                {
                                    "operation": operation,
                                    "backend": backend_name,
                                    "array_shape": shape_label(args.array_shape),
                                    "chunks": shape_label(chunks),
                                    "shards": shape_label(shards),
                                    "repeat": repeat,
                                    "seconds": seconds,
                                    "mib_per_second": total_bytes / (1024**2 * seconds),
                                    "validated": True,
                                }
                            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-shape", type=parse_shape, default=DEFAULT_ARRAY_SHAPE)
    parser.add_argument("--chunks", type=parse_shape, nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--shards", type=parse_optional_shape, nargs="+", default=DEFAULT_SHARDS
    )
    parser.add_argument(
        "--operations", nargs="+", choices=("read", "write"), default=("read", "write")
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=non_negative_int, default=1)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    """Run the benchmark and write CSV results."""
    args = build_parser().parse_args()
    write_results(run_benchmark(args), RESULT_FIELDS, args.output)


if __name__ == "__main__":
    main()
