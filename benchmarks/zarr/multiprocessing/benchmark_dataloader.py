"""Benchmark CAREamics Zarr loading with PyTorch DataLoader workers.

Run from the repository root, for example::

    uv run python benchmarks/zarr/multiprocessing/benchmark_dataloader.py \
        --backends zarr,zarrs,tensorstore \
        --workers 0,1,2,4 \
        --repeats 3

Each benchmark case runs in a subprocess so that a worker crash or deadlock does not
terminate the complete benchmark sweep.
"""

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
CONTEXTS = ("auto", "fork", "forkserver", "spawn")
DEFAULT_SHAPE = (8, 1, 1024, 1024)
DEFAULT_CHUNKS = (1, 1, 64, 64)
DEFAULT_OUTPUT = Path("benchmarks/zarr/multiprocessing/results.csv")
SEED = 24

RESULT_FIELDS = (
    "backend",
    "workers",
    "requested_context",
    "effective_context",
    "persistent_workers",
    "repeat",
    "status",
    "dataset_build_seconds",
    "first_batch_seconds",
    "measured_batches",
    "steady_state_seconds",
    "batches_per_second",
    "samples_per_second",
    "batch_shape",
    "error",
)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of non-negative integers.

    Parameters
    ----------
    value : str
        Comma-separated integer values.

    Returns
    -------
    list of int
        Parsed values.
    """
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}.") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"Expected non-negative integers: {value!r}.")
    return values


def parse_backend_list(value: str) -> list[str]:
    """Parse and validate a comma-separated backend list.

    Parameters
    ----------
    value : str
        Comma-separated backend names.

    Returns
    -------
    list of str
        Validated backend names.
    """
    values = [item.strip() for item in value.split(",")]
    invalid = sorted(set(values).difference(BACKENDS))
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported backend(s): {', '.join(invalid)}."
        )
    return values


def parse_shape(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive shape.

    Parameters
    ----------
    value : str
        Comma-separated dimensions.

    Returns
    -------
    tuple of int
        Parsed shape.
    """
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid shape: {value!r}.") from error
    if not shape or any(size <= 0 for size in shape):
        raise argparse.ArgumentTypeError(f"Expected positive dimensions: {value!r}.")
    return shape


def create_store(
    store_path: Path, shape: tuple[int, ...], chunks: tuple[int, ...]
) -> None:
    """Create a deterministic Zarr v3 array for the benchmark.

    Parameters
    ----------
    store_path : pathlib.Path
        Destination Zarr store.
    shape : tuple of int
        Array shape in SCYX order.
    chunks : tuple of int
        Zarr chunk shape.
    """
    if len(shape) != 4 or len(chunks) != 4:
        raise ValueError(
            "The benchmark currently expects a four-dimensional SCYX array."
        )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = rng.integers(0, 256, size=shape, dtype=np.uint16)
    zarr.create_array(
        store=store_path,
        data=data,
        chunks=chunks,
        zarr_format=3,
        overwrite=True,
    )


def build_dataset(
    store_path: Path,
    backend: str,
    workers: int,
    batch_size: int,
    patch_size: int,
) -> tuple[Any, Any]:
    """Build the CAREamics dataset used by one benchmark case.

    Parameters
    ----------
    store_path : pathlib.Path
        Input Zarr store.
    backend : str
        CAREamics Zarr backend name.
    workers : int
        Number of DataLoader workers.
    batch_size : int
        DataLoader batch size.
    patch_size : int
        Spatial patch size.

    Returns
    -------
    tuple
        DataConfig and CareamicsDataset instances.
    """
    from careamics.config.data.data_config import DataConfig
    from careamics.dataset.factory import create_dataset

    config = DataConfig(
        mode="training",
        data_type="zarr",
        axes="SCYX",
        patching={
            "name": "random",
            "patch_size": [patch_size, patch_size],
            "seed": SEED,
        },
        normalization={"name": "none"},
        batch_size=batch_size,
        in_memory=False,
        num_workers=workers,
        zarr_backend=backend,
        train_dataloader_params={"shuffle": False, "num_workers": workers},
    )
    return config, create_dataset(config, inputs=[store_path], targets=None)


def effective_context(requested: str, backend: str, workers: int) -> str:
    """Return the context expected for a benchmark case.

    Parameters
    ----------
    requested : str
        Requested multiprocessing context.
    backend : str
        CAREamics Zarr backend name.
    workers : int
        Number of DataLoader workers.

    Returns
    -------
    str
        Effective context label.
    """
    if workers == 0:
        return "none"
    if requested != "auto":
        return requested
    if backend == "tensorstore" and sys.platform == "linux":
        return "forkserver"
    return mp.get_start_method()


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one benchmark case inside a worker subprocess.

    Parameters
    ----------
    case : dict
        Serialized benchmark case parameters.

    Returns
    -------
    dict
        Benchmark metrics and status.
    """
    backend = case["backend"]
    workers = case["workers"]
    requested_context = case["context"]
    persistent_workers = case["persistent_workers"]

    if workers > 0 and requested_context != "auto":
        if requested_context not in mp.get_all_start_methods():
            return {
                **case,
                "effective_context": requested_context,
                "status": "skipped",
                "error": "multiprocessing context is unavailable",
            }

    try:
        start = time.perf_counter()
        config, dataset = build_dataset(
            store_path=Path(case["store"]),
            backend=backend,
            workers=workers,
            batch_size=case["batch_size"],
            patch_size=case["patch_size"],
        )
        dataset_build_seconds = time.perf_counter() - start

        from torch.utils.data import DataLoader
        from torch.utils.data._utils.collate import default_collate

        loader_kwargs: dict[str, Any] = {
            **config.train_dataloader_params,
            "batch_size": config.batch_size,
            "collate_fn": default_collate,
            "num_workers": workers,
            "persistent_workers": persistent_workers if workers > 0 else False,
        }
        if workers > 0 and requested_context != "auto":
            loader_kwargs["multiprocessing_context"] = requested_context

        first_batch_start = time.perf_counter()
        loader = DataLoader(dataset, **loader_kwargs)
        iterator = iter(loader)
        first_batch = next(iterator)
        first_batch_seconds = time.perf_counter() - first_batch_start

        for _ in range(case["warmup_batches"]):
            next(iterator)

        measured_batches = case["measure_batches"]
        measure_start = time.perf_counter()
        for _ in range(measured_batches):
            next(iterator)
        steady_state_seconds = time.perf_counter() - measure_start

        batch_data = first_batch[0].data
        batch_size = int(batch_data.shape[0])
        return {
            **case,
            "effective_context": effective_context(requested_context, backend, workers),
            "status": "ok",
            "dataset_build_seconds": dataset_build_seconds,
            "first_batch_seconds": first_batch_seconds,
            "measured_batches": measured_batches,
            "steady_state_seconds": steady_state_seconds,
            "batches_per_second": measured_batches / steady_state_seconds,
            "samples_per_second": measured_batches * batch_size / steady_state_seconds,
            "batch_shape": tuple(int(size) for size in batch_data.shape),
            "error": "",
        }
    except StopIteration as error:
        raise RuntimeError(
            "The dataset ended before the requested number of benchmark batches. "
            "Reduce --warmup-batches/--measure-batches or increase the store size."
        ) from error
    except BaseException as error:  # subprocess boundary: report crashes cleanly
        traceback.print_exc()
        return {
            **case,
            "effective_context": effective_context(requested_context, backend, workers),
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def run_case_subprocess(case: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Run one case in a fresh Python process with a timeout.

    Parameters
    ----------
    case : dict
        Serialized benchmark case parameters.
    timeout : float
        Maximum runtime in seconds.

    Returns
    -------
    dict
        Benchmark metrics and status.
    """
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
            "effective_context": effective_context(
                case["context"], case["backend"], case["workers"]
            ),
            "status": "timeout",
            "error": f"case exceeded {timeout:g} seconds: {error}",
        }
    if completed.returncode == 0:
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass

    status = "timeout" if completed.returncode is None else "crash"
    error = completed.stderr.strip() or completed.stdout.strip()
    return {
        **case,
        "effective_context": effective_context(
            case["context"], case["backend"], case["workers"]
        ),
        "status": status,
        "error": error or f"subprocess exited with code {completed.returncode}",
    }


def write_results(rows: list[dict[str, Any]], output: Path | None) -> None:
    """Write benchmark results as CSV or to standard output.

    Parameters
    ----------
    rows : list of dict
        Benchmark result rows.
    output : pathlib.Path or None
        Output CSV path, or ``None`` for standard output.
    """
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    stream = sys.stdout if output is None else output.open("w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            [
                {
                    **row,
                    "requested_context": row.get("requested_context", row["context"]),
                }
                for row in rows
            ]
        )
    finally:
        if output is not None:
            stream.close()


def run_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Create the input store and run the requested benchmark matrix.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list of dict
        Benchmark result rows.
    """
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.store is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="careamics-zarr-")
        store_path = Path(temporary_directory.name) / "input.zarr"
        create_store(store_path, args.shape, args.chunks)
    else:
        store_path = args.store
        if not store_path.exists():
            create_store(store_path, args.shape, args.chunks)

    rows: list[dict[str, Any]] = []
    try:
        for backend in args.backends:
            for workers in args.workers:
                contexts = ("auto",) if workers == 0 else args.contexts
                for context in contexts:
                    for repeat in range(args.repeats):
                        case = {
                            "backend": backend,
                            "workers": workers,
                            "context": context,
                            "persistent_workers": args.persistent_workers,
                            "repeat": repeat,
                            "store": str(store_path),
                            "batch_size": args.batch_size,
                            "patch_size": args.patch_size,
                            "warmup_batches": args.warmup_batches,
                            "measure_batches": args.measure_batches,
                        }
                        print(
                            f"{backend=} {workers=} {context=} {repeat=}",
                            file=sys.stderr,
                        )
                        rows.append(run_case_subprocess(case, args.timeout))
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed benchmark options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=parse_backend_list, default=list(BACKENDS))
    parser.add_argument("--workers", type=parse_int_list, default=[0, 1, 2, 4])
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=["auto"])
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--shape", type=parse_shape, default=DEFAULT_SHAPE)
    parser.add_argument("--chunks", type=parse_shape, default=DEFAULT_CHUNKS)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Run either one subprocess case or the complete benchmark sweep.

    Returns
    -------
    None
        Results are written to the configured CSV output.
    """
    args = parse_args()
    if args.case:
        print(json.dumps(run_case(json.load(sys.stdin)), default=str))
        return

    if args.repeats <= 0 or args.warmup_batches < 0 or args.measure_batches <= 0:
        raise ValueError("repeats must be positive; batch counts are invalid")
    if len(args.shape) != len(args.chunks):
        raise ValueError("--shape and --chunks must have the same rank")

    rows = run_sweep(args)
    output = None if str(args.output) == "-" else args.output
    write_results(rows, output)


if __name__ == "__main__":
    main()
