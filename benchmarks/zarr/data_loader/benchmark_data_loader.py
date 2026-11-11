#!/usr/bin/env python
"""Benchmark CAREamics random patch loading through PyTorch DataLoader workers."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import zarr

BACKENDS = ("zarr", "zarrs", "tensorstore")
WORKERS = (0, 1, 2, 3, 4)
ARRAY_SHAPE = (8, 1, 1024, 1024)
CHUNKS = (1, 1, 128, 128)
SHARDS = (1, 1, 512, 512)
PATCH_SIZE = 128
PATCHES = 128
BATCH_SIZE = 8
WARMUP_BATCHES = 1
REPEATS = 3
SEED = 24
DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "results" / "data_loader" / "data_loader_results.csv"
)
RESULT_FIELDS = (
    "backend",
    "workers",
    "repeat",
    "context",
    "patches",
    "batch_size",
    "patch_size",
    "first_batch_seconds",
    "total_seconds",
    "patches_per_second",
    "mib_per_second",
    "samples_per_second",
    "status",
    "error",
)


def create_store(path: Path) -> None:
    """Create the fixed sharded Zarr input store."""
    rng = np.random.default_rng(SEED)
    zarr.create_array(
        store=path,
        data=rng.integers(0, 256, size=ARRAY_SHAPE, dtype=np.uint16),
        chunks=CHUNKS,
        shards=SHARDS,
        zarr_format=3,
        overwrite=True,
    )


def worker_context(backend: str, workers: int) -> str:
    """Select a fresh-process context for worker cases."""
    if workers == 0:
        return "none"
    available = mp.get_all_start_methods()
    if "forkserver" in available and sys.platform == "linux":
        return "forkserver"
    if "spawn" in available:
        return "spawn"
    return mp.get_start_method()


def build_dataset(store: Path, backend: str, workers: int) -> Any:
    """Build a CAREamics dataset using random patching."""
    from careamics.config.data.data_config import DataConfig
    from careamics.dataset.factory import create_dataset

    config = DataConfig(
        mode="training",
        data_type="zarr",
        axes="SCYX",
        patching={
            "name": "random",
            "patch_size": [PATCH_SIZE, PATCH_SIZE],
            "seed": SEED,
        },
        normalization={"name": "none"},
        batch_size=BATCH_SIZE,
        in_memory=False,
        num_workers=workers,
        zarr_backend=backend,
        train_dataloader_params={"shuffle": False},
    )
    dataset = create_dataset(config, inputs=[store], targets=None)
    required_patches = PATCHES + WARMUP_BATCHES * BATCH_SIZE
    if len(dataset) < required_patches:
        raise RuntimeError(
            f"Dataset contains {len(dataset)} patches; {required_patches} are required."
        )
    return dataset, config


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one backend/worker case inside a fresh Python process."""
    backend = case["backend"]
    workers = case["workers"]
    context = worker_context(backend, workers)
    result = {
        **case,
        "context": context,
        "status": "error",
        "error": "",
    }
    try:
        from torch.utils.data import DataLoader, Subset
        from torch.utils.data._utils.collate import default_collate

        dataset, config = build_dataset(Path(case["store"]), backend, workers)
        subset = Subset(dataset, range(PATCHES + WARMUP_BATCHES * BATCH_SIZE))
        loader_kwargs: dict[str, Any] = {
            "batch_size": config.batch_size,
            "shuffle": False,
            "collate_fn": default_collate,
            "num_workers": workers,
        }
        if workers > 0:
            loader_kwargs["multiprocessing_context"] = context

        started = time.perf_counter()
        loader = DataLoader(subset, **loader_kwargs)
        iterator = iter(loader)
        next(iterator)
        first_batch_seconds = time.perf_counter() - started
        measured_started = time.perf_counter()
        for _ in iterator:
            pass
        total_seconds = time.perf_counter() - measured_started

        result.update(
            {
                "first_batch_seconds": first_batch_seconds,
                "total_seconds": total_seconds,
                "patches_per_second": PATCHES / total_seconds,
                "samples_per_second": PATCHES / total_seconds,
                "mib_per_second": (
                    PATCHES * PATCH_SIZE * PATCH_SIZE * np.dtype(np.uint16).itemsize
                )
                / (1024**2 * total_seconds),
                "status": "ok",
            }
        )
        del loader
    except BaseException as error:
        traceback.print_exc()
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def run_case_subprocess(case: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Run a case in a subprocess and report crashes or timeouts."""
    command = [sys.executable, str(Path(__file__).resolve()), "--case"]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(case),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            **case,
            "context": worker_context(case["backend"], case["workers"]),
            "status": "timeout",
            "error": str(error),
        }
    if completed.returncode == 0:
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
    return {
        **case,
        "context": worker_context(case["backend"], case["workers"]),
        "status": "crash",
        "error": completed.stderr.strip() or completed.stdout.strip(),
    }


def write_results(rows: list[dict[str, Any]], output: Path) -> None:
    """Write the benchmark rows to CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run all backend/worker cases with the fixed benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(run_case(json.load(sys.stdin))))
        return

    with tempfile.TemporaryDirectory(prefix="careamics-zarr-dataloader-") as directory:
        store = Path(directory) / "input.zarr"
        create_store(store)
        rows: list[dict[str, Any]] = []
        for backend in BACKENDS:
            for workers in WORKERS:
                for repeat in range(1, REPEATS + 1):
                    case = {
                        "backend": backend,
                        "workers": workers,
                        "repeat": repeat,
                        "patches": PATCHES,
                        "batch_size": BATCH_SIZE,
                        "patch_size": PATCH_SIZE,
                        "store": str(store),
                    }
                    print(
                        f"Running backend={backend}, workers={workers}, "
                        f"repeat={repeat}",
                        file=sys.stderr,
                    )
                    rows.append(run_case_subprocess(case, args.timeout))
    write_results(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
