"""Run a CAREamics I/O screening matrix using the existing benchmark runners.

The patch and tile benchmarks cover chunk/shard layouts. The worker benchmark is
run once per patch/chunk combination because its runner models a single dataset
layout per process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
IO_DIR = ROOT / "io"
MULTIPROCESSING_DIR = ROOT / "multiprocessing"


def parse_list(value: str) -> list[str]:
    """Parse a comma-separated non-empty list."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return values


def run(command: list[str]) -> bool:
    """Run one child benchmark and stream its output."""
    print("Running:", " ".join(command), file=sys.stderr)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"Benchmark failed with exit code {completed.returncode}.", file=sys.stderr
        )
        return False
    return True


def shape_arg(value: str) -> tuple[int, int]:
    """Parse a square XY size or a comma-separated 2D shape."""
    parts = tuple(int(part) for part in value.replace("x", ",").split(","))
    if len(parts) == 1:
        parts = (parts[0], parts[0])
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(f"Expected a positive 2D shape: {value!r}.")
    return parts


def square_shape_args(values: list[str], name: str) -> list[str]:
    """Normalize square shape labels for child parsers."""
    normalized: list[str] = []
    for value in values:
        height, width = shape_arg(value)
        if height != width:
            raise argparse.ArgumentTypeError(
                f"{name} currently supports square values only: {value!r}."
            )
        normalized.append(f"{height},{width}")
    return normalized


def main() -> None:
    """Run patch-read, tile-write, and worker-scaling benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "screening"
    )
    parser.add_argument("--array-shape", type=shape_arg, default=(2048, 2048))
    parser.add_argument(
        "--patch-sizes", type=parse_list, default=["64x64", "128x128", "256x256"]
    )
    parser.add_argument("--tile-sizes", type=parse_list, default=["128x128", "256x256"])
    parser.add_argument(
        "--chunks", type=parse_list, default=["64x64", "128x128", "256x256"]
    )
    parser.add_argument(
        "--shards", type=parse_list, default=["none", "256x256", "512x512"]
    )
    parser.add_argument(
        "--backends", type=parse_list, default=["zarr", "zarrs", "tensorstore"]
    )
    parser.add_argument("--workers", type=parse_list, default=["0", "2", "4"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=100)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = f"{args.array_shape[0]},{args.array_shape[1]}"
    backends = ",".join(args.backends)
    patch_sizes = square_shape_args(args.patch_sizes, "patch sizes")
    tile_sizes = square_shape_args(args.tile_sizes, "tile sizes")
    chunks = square_shape_args(args.chunks, "chunks")
    # Preserve the requested shard order while normalizing non-None values.
    normalized_shards = []
    non_none_index = 0
    normalized_non_none = square_shape_args(
        [value for value in args.shards if value != "none"], "shards"
    )
    for value in args.shards:
        if value == "none":
            normalized_shards.append("none")
        else:
            normalized_shards.append(normalized_non_none[non_none_index])
            non_none_index += 1
    shards = normalized_shards

    run_records: list[dict[str, object]] = []

    def run_and_record(command: list[str], output: Path) -> bool:
        """Run a child, record its status, and prevent stale output reuse."""
        output.unlink(missing_ok=True)
        success = run(command)
        run_records.append(
            {"output": str(output), "success": success, "command": command}
        )
        return success

    read_results = args.output_dir / "patch_read_results.csv"
    run_and_record(
        [
            sys.executable,
            str(IO_DIR / "benchmark_read_patch.py"),
            "--array-shape",
            shape,
            "--patch-sizes",
            *patch_sizes,
            "--chunks",
            *chunks,
            "--shards",
            *shards,
            "--backends",
            *args.backends,
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--output",
            str(read_results),
        ],
        read_results,
    )

    write_results = args.output_dir / "tile_write_results.csv"
    run_and_record(
        [
            sys.executable,
            str(IO_DIR / "benchmark_write_tile.py"),
            "--array-shape",
            shape,
            "--tile-sizes",
            *tile_sizes,
            "--chunks",
            *chunks,
            "--shards",
            *shards,
            "--backends",
            *args.backends,
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--output",
            str(write_results),
        ],
        write_results,
    )

    worker_dir = args.output_dir / "worker_results"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_results: list[str] = []
    for patch_size in patch_sizes:
        patch_value = patch_size.split(",")[0]
        for chunk in chunks:
            chunk_value = chunk.split(",")[0]
            output = worker_dir / f"patch_{patch_value}_chunk_{chunk_value}.csv"
            run_and_record(
                [
                    sys.executable,
                    str(MULTIPROCESSING_DIR / "benchmark_dataloader.py"),
                    "--backends",
                    backends,
                    "--workers",
                    ",".join(args.workers),
                    "--contexts",
                    "auto",
                    "--repeats",
                    str(args.repeats),
                    "--warmup-batches",
                    str(args.warmups),
                    "--measure-batches",
                    str(args.measure_batches),
                    "--shape",
                    f"8,1,{args.array_shape[0]},{args.array_shape[1]}",
                    "--chunks",
                    f"1,1,{chunk_value},{chunk_value}",
                    "--patch-size",
                    patch_value,
                    "--output",
                    str(output),
                ],
                output,
            )
            worker_results.append(str(output))

    manifest = {
        "patch_read": str(read_results),
        "tile_write": str(write_results),
        "worker_results": worker_results,
        "runs": run_records,
        "note": "Worker runs use unsharded stores; patch/tile runs cover shards.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
