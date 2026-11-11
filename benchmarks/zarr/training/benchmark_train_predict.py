#!/usr/bin/env python
"""Benchmark light-sheet training and prediction for each Zarr backend."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pooch

from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config

NUM_EPOCHS = 20
DATA_URL = "https://zenodo.org/records/22078388/files/001-small-lowT.zip?download=1"
DATA_HASH = "5069b9606d646f810e9f0b899d98916d8366bfb14bc74765774c1d4df4a5fe80"


def main() -> None:
    """Run the fixed 20-epoch benchmark and write a CSV report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results" / "train_predict",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("zarr", "zarrs", "tensorstore"),
        default=("zarr", "zarrs", "tensorstore"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or args.output_dir / "lightsheet_training_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    data_files = pooch.retrieve(
        DATA_URL,
        known_hash=DATA_HASH,
        processor=pooch.Unzip(),
        path=Path(__file__).parents[3] / "examples" / "ome-ngff" / "data",
    )
    data_path = next(
        Path(file).parent
        for file in data_files
        if file.endswith("raw.ome.zarr/zarr.json")
    )

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
            run_dir = args.output_dir / backend / f"repeat_{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"Running backend={backend}, repeat={repeat}", flush=True)
            row: dict[str, object] = {
                "backend": backend,
                "repeat": repeat,
                "num_epochs": NUM_EPOCHS,
                "training_seconds": "",
                "prediction_seconds": "",
                "work_dir": str(run_dir),
                "checkpoint": "",
                "status": "error",
                "error": "",
            }
            try:
                config = create_advanced_n2v_config(
                    experiment_name="n2v_lightsheet_benchmark",
                    data_type="zarr",
                    axes="TCZYX",
                    patch_size=(32, 128, 128),
                    batch_size=6,
                    num_epochs=NUM_EPOCHS,
                    n_channels=2,
                    independent_channels=True,
                    zarr_backend=backend,
                    num_workers=0,
                )
                careamist = CAREamist(config=config, work_dir=run_dir)

                started = time.perf_counter()
                careamist.train(train_data=data_path)
                row["training_seconds"] = time.perf_counter() - started

                checkpoints = careamist.get_checkpoints()
                if not checkpoints:
                    raise RuntimeError(f"No checkpoint was produced in {run_dir}")
                row["checkpoint"] = str(checkpoints[-1])

                started = time.perf_counter()
                careamist.predict_to_disk(
                    pred_data=data_path,
                    prediction_dir=run_dir / "predictions",
                    batch_size=16,
                    tile_size=(32, 128, 128),
                    tile_overlap=(4, 48, 48),
                )
                row["prediction_seconds"] = time.perf_counter() - started
                row["status"] = "ok"
            except Exception as error:
                row["error"] = repr(error)
                print(f"Failed: {error}", flush=True)
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
