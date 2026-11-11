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
    Path(__file__).parents[1] / "result" / "data_loader" / "data_loader_results.csv"
)
DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "result" / "data_loader" / "data_loader.png"
)


def main() -> None:
    """Plot mean MiB/s with standard-deviation bands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"No successful benchmark rows found in {args.input}.")

    values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(row["backend"], row["workers"])].append(float(row["mib_per_second"]))
    backends = sorted({backend for backend, _ in values})
    workers = sorted({int(worker) for _, worker in values})
    figure, axis = plt.subplots(figsize=(8, 5))
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
    axis.set_ylabel("Mean speed (MiB/s) ± SD")
    axis.set_title("CAREamics random-patch DataLoader benchmark")
    axis.set_xticks(workers)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
