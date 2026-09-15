#!/usr/bin/env python
"""Benchmark N2V training and prediction on a 3D OME-NGFF light-sheet image.

Data from Hess, M., Caton, M., Kothari, M., Swedlow, J., Liberali, P., & Quintas
Glasner de Medeiros, G. (2026). Reference OME-Zarr for 3D time-lapse light-sheet
microscopy with nuclei tracking (Version v0.1) [Dataset]. Zenodo.
https://doi.org/10.5281/zenodo.22078388
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter
from typing import Literal

import pooch

from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from careamics.plotting import plot_loss

BENCHMARK_DIR = Path(__file__).parent
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"
ZARR_BACKENDS = ("zarr", "zarrs", "tensorstore")
ZarrBackend = Literal["zarr", "zarrs", "tensorstore"]
TIMING_FIELDS = (
    "zarr_backend",
    "training_seconds",
    "prediction_seconds",
    "total_seconds",
)


def download_data() -> Path:
    """Download the light-sheet benchmark data.

    Returns
    -------
    pathlib.Path
        Path to the downloaded OME-Zarr store.
    """
    file_list = pooch.retrieve(
        "https://zenodo.org/records/22078388/files/001-small-lowT.zip?download=1",
        known_hash=("5069b9606d646f810e9f0b899d98916d8366bfb14bc74765774c1d4df4a5fe80"),
        processor=pooch.Unzip(),
        path=BENCHMARK_DIR / "data",
    )

    for file_path in file_list:
        if file_path.endswith("raw.ome.zarr/zarr.json"):
            return Path(file_path).parent

    raise FileNotFoundError("Downloaded archive does not contain raw.ome.zarr.")


def write_timings(
    results_dir: Path,
    zarr_backend: ZarrBackend,
    training_seconds: float,
    prediction_seconds: float,
) -> Path:
    """Write benchmark timings to a new incrementally named CSV.

    Parameters
    ----------
    results_dir : pathlib.Path
        Directory in which to create the CSV.
    zarr_backend : {"zarr", "zarrs", "tensorstore"}
        Zarr backend used by the benchmark.
    training_seconds : float
        Training duration in seconds.
    prediction_seconds : float
        Prediction duration in seconds.

    Returns
    -------
    pathlib.Path
        Path to the created CSV.
    """
    results_dir.mkdir(exist_ok=True, parents=True)
    index = 0
    while True:
        output = results_dir / f"result_{zarr_backend}_{index}.csv"
        try:
            with output.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "zarr_backend": zarr_backend,
                        "training_seconds": training_seconds,
                        "prediction_seconds": prediction_seconds,
                        "total_seconds": training_seconds + prediction_seconds,
                    }
                )
        except FileExistsError:
            index += 1
            continue
        return output


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-backend",
        choices=ZARR_BACKENDS,
        default="zarr",
        help="Zarr backend used for training, prediction, and writing.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main() -> None:
    """Run and time light-sheet training and prediction.

    Returns
    -------
    None
        Benchmark timings are written to CSV.
    """
    args = build_parser().parse_args()
    zarr_backend: ZarrBackend = args.zarr_backend

    root = BENCHMARK_DIR / "n2v_training"
    root.mkdir(exist_ok=True, parents=True)

    path_to_zarr = download_data()

    config = create_advanced_n2v_config(
        experiment_name="n2v_lightsheet_denoising",
        data_type="zarr",
        axes="TCZYX",
        patch_size=(32, 128, 128),
        batch_size=6,
        num_epochs=50,
        n_channels=2,
        independent_channels=True,
        zarr_backend=zarr_backend,
        num_workers=0,
        # train_dataloader_params={
        #     "multiprocessing_context": 'spawn'
        # },
        # val_dataloader_params={
        #     "multiprocessing_context": 'spawn'
        # },
        seed=24,
    )
    careamist = CAREamist(config=config, work_dir=root)

    training_start = perf_counter()
    careamist.train(train_data=path_to_zarr)
    training_seconds = perf_counter() - training_start

    results = root / "results"
    results.mkdir(exist_ok=True, parents=True)
    plot_loss(careamist.get_losses(), save_path=results / "loss.png")

    prediction_start = perf_counter()
    careamist.predict_to_disk(
        pred_data=path_to_zarr,
        batch_size=16,
        tile_size=(32, 128, 128),
        tile_overlap=(4, 48, 48),
        num_workers=0,
    )
    prediction_seconds = perf_counter() - prediction_start

    output = write_timings(
        args.results_dir, zarr_backend, training_seconds, prediction_seconds
    )
    print(f"Training: {training_seconds:.3f} s")
    print(f"Prediction: {prediction_seconds:.3f} s")
    print(f"Total: {training_seconds + prediction_seconds:.3f} s")
    print(f"Timings written to {output}")


if __name__ == "__main__":
    main()
