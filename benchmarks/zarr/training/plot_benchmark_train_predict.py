"""Plot light-sheet training and prediction benchmark results."""

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
    Path(__file__).parents[1]
    / "result"
    / "train_predict"
    / "lightsheet_training_results.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "result" / "train_predict" / "train_predict.png"
)


def main() -> None:
    """Plot mean training and prediction time with standard-deviation bars."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"No successful benchmark rows found in {args.input}.")

    values: defaultdict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"training_seconds": [], "prediction_seconds": []}
    )
    for row in rows:
        for metric in ("training_seconds", "prediction_seconds"):
            values[row["backend"]][metric].append(float(row[metric]))

    backends = sorted(values)
    metrics = ("training_seconds", "prediction_seconds")
    means = [
        [statistics.mean(values[backend][metric]) for backend in backends]
        for metric in metrics
    ]
    deviations = [
        [
            (
                statistics.stdev(values[backend][metric])
                if len(values[backend][metric]) > 1
                else 0.0
            )
            for backend in backends
        ]
        for metric in metrics
    ]

    positions = range(len(backends))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        [position - width / 2 for position in positions],
        means[0],
        width,
        yerr=deviations[0],
        capsize=4,
        label="Training",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        means[1],
        width,
        yerr=deviations[1],
        capsize=4,
        label="Prediction",
    )
    axis.set_xticks(list(positions), backends)
    axis.set_ylabel("Mean time (s) ± SD")
    axis.set_title("Light-sheet training and prediction benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
