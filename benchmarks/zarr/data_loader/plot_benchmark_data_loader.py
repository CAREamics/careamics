"""Plot DataLoader worker-scaling benchmark results."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = (
    Path(__file__).parents[1] / "results" / "data_loader" / "data_loader_results.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "results" / "data_loader" / "data_loader.png"
)


def _plot_mean_and_deviation(
    axis: plt.Axes,
    values: defaultdict[tuple[str, str], list[float]],
    backends: list[str],
    workers: list[int],
    ylabel: str,
) -> None:
    """Plot one metric as a mean curve with standard-deviation bands."""
    for backend in backends:
        means = [statistics.mean(values[(backend, str(worker))]) for worker in workers]
        deviations = [
            (
                statistics.stdev(values[(backend, str(worker))])
                if len(values[(backend, str(worker))]) > 1
                else 0.0
            )
            for worker in workers
        ]
        axis.plot(workers, means, marker="o", label=backend)
        axis.fill_between(
            workers,
            [
                mean - deviation
                for mean, deviation in zip(means, deviations, strict=True)
            ],
            [
                mean + deviation
                for mean, deviation in zip(means, deviations, strict=True)
            ],
            alpha=0.15,
        )
    axis.set_xlabel("DataLoader workers")
    axis.set_ylabel(ylabel)
    axis.set_xticks(workers)
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    """Plot steady-state speed and first-batch startup time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"No successful benchmark rows found in {args.input}.")

    speed_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    startup_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["backend"], row["workers"])
        speed_values[key].append(float(row["mib_per_second"]))
        if row.get("first_batch_seconds"):
            startup_values[key].append(float(row["first_batch_seconds"]))
    backends = sorted({backend for backend, _ in speed_values})
    workers = sorted({int(worker) for _, worker in speed_values})
    figure, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    _plot_mean_and_deviation(
        axes[0], speed_values, backends, workers, "Mean speed (MiB/s) ± SD"
    )
    axes[0].set_title("CAREamics random-patch DataLoader benchmark")
    axes[0].legend()

    if startup_values:
        startup_backends = sorted({backend for backend, _ in startup_values})
        startup_workers = sorted({int(worker) for _, worker in startup_values})
        _plot_mean_and_deviation(
            axes[1],
            startup_values,
            startup_backends,
            startup_workers,
            "Mean startup time (s) ± SD",
        )
        axes[1].set_title("DataLoader startup and first-batch time")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
