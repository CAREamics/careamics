#!/usr/bin/env python
"""Benchmark light-sheet training and prediction for each Zarr backend."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import time
from pathlib import Path
from types import ModuleType

from careamics import CAREamist


def load_example() -> ModuleType:
    """Load the neighboring light-sheet example as a module."""
    path = Path(__file__).with_name("n2v_lightsheet_denoising.py")
    spec = importlib.util.spec_from_file_location("lightsheet_example", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load light-sheet example from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(
    example: ModuleType,
    backend: str,
    repeat: int,
    output_dir: Path,
    num_epochs: int,
) -> dict[str, object]:
    """Run one backend/repeat and time training and prediction independently."""
    run_dir = output_dir / backend / f"repeat_{repeat}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = example.download_data()
    careamist = CAREamist(
        config=example.create_config(backend=backend, num_epochs=num_epochs),
        work_dir=run_dir,
    )

    started = time.perf_counter()
    careamist.train(train_data=data_path)
    training_seconds = time.perf_counter() - started

    checkpoints = careamist.get_checkpoints()
    if not checkpoints:
        raise RuntimeError(f"No checkpoint was produced in {run_dir}")

    prediction_dir = run_dir / "predictions"
    started = time.perf_counter()
    careamist.predict_to_disk(
        pred_data=data_path,
        prediction_dir=prediction_dir,
        batch_size=16,
        tile_size=(32, 128, 128),
        tile_overlap=(4, 48, 48),
    )
    prediction_seconds = time.perf_counter() - started

    return {
        "backend": backend,
        "repeat": repeat,
        "num_epochs": num_epochs,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "work_dir": str(run_dir),
        "checkpoint": str(checkpoints[-1]),
        "status": "ok",
        "error": "",
    }


def main() -> None:
    """Run all backend/repeat cases and write a CSV report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("lightsheet_benchmark"))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("zarr", "zarrs", "tensorstore"),
        default=("zarr", "zarrs", "tensorstore"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1 or args.num_epochs < 1:
        raise ValueError("repeats and num-epochs must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or args.output_dir / "lightsheet_training_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    example = load_example()
    fields = (
        "backend",
        "repeat",
        "num_epochs",
        "training_seconds",
        "prediction_seconds",
        "work_dir",
        "checkpoint",
        "status",
        "error",
    )
    rows: list[dict[str, object]] = []
    for backend in args.backends:
        for repeat in range(1, args.repeats + 1):
            print(f"Running backend={backend}, repeat={repeat}", flush=True)
            try:
                row = run_case(
                    example, backend, repeat, args.output_dir, args.num_epochs
                )
            except Exception as error:
                row = {
                    "backend": backend,
                    "repeat": repeat,
                    "num_epochs": args.num_epochs,
                    "training_seconds": "",
                    "prediction_seconds": "",
                    "work_dir": str(args.output_dir / backend / f"repeat_{repeat}"),
                    "checkpoint": "",
                    "status": "error",
                    "error": repr(error),
                }
                print(f"Failed: {error}", flush=True)
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
