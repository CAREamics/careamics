"""Summarize and plot CAREamics whole-image write benchmark results."""

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
    / "prediction_writer"
    / "plane_write_results.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "result" / "prediction_writer"
Stats = tuple[float, float]
BACKEND_COLORS = {
    "tensorstore": "#1f77b4",  # blue
    "zarrs": "#2ca02c",  # green
    "zarr": "#ff7f0e",  # orange
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate image-write benchmark rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}.")
    failures = [row for row in rows if row.get("validated") != "True"]
    if failures:
        details = ", ".join(row.get("error_type", "unknown") for row in failures[:3])
        raise ValueError(f"Found {len(failures)} failed rows ({details}).")
    return rows


def stats_by_configuration(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], Stats]:
    """Aggregate repeats into mean throughput and sample standard deviation."""
    values: defaultdict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["backend"], row["chunks"], row["shards"])
        values[key].append(float(row["mib_per_second"]))
    return {
        key: (
            statistics.mean(samples),
            statistics.stdev(samples) if len(samples) > 1 else 0.0,
        )
        for key, samples in values.items()
    }


def sort_shapes(values: set[str]) -> list[str]:
    """Sort shape labels numerically."""
    return sorted(
        values, key=lambda value: tuple(int(part) for part in value.split("x"))
    )


def plot_results(
    stats: dict[tuple[str, str, str], Stats],
    rows: list[dict[str, str]],
    output: Path,
    dpi: int,
) -> None:
    """Plot mean image-write throughput with standard-deviation bands."""
    shards = sorted({key[2] for key in stats})
    backends = sorted({key[0] for key in stats})
    chunks = sort_shapes({key[1] for key in stats})
    colors = {backend: BACKEND_COLORS.get(backend, "#7f7f7f") for backend in backends}
    x = range(len(chunks))
    fig, axes = plt.subplots(
        1,
        len(shards),
        figsize=(max(6.5, 4.5 * len(shards)), 4.5),
        sharey=True,
        squeeze=False,
    )
    for axis, shard in zip(axes.flat, shards, strict=True):
        for backend in backends:
            entries = [stats.get((backend, chunk, shard)) for chunk in chunks]
            if any(entry is None for entry in entries):
                continue
            means = [
                entry[0] if entry is not None else float("nan") for entry in entries
            ]
            deviations = [entry[1] if entry is not None else 0.0 for entry in entries]
            axis.plot(
                x,
                means,
                marker="o",
                color=colors[backend],
                label=backend,
            )
            axis.fill_between(
                x,
                [
                    value - deviation
                    for value, deviation in zip(means, deviations, strict=True)
                ],
                [
                    value + deviation
                    for value, deviation in zip(means, deviations, strict=True)
                ],
                color=colors[backend],
                alpha=0.12,
            )
        axis.set_title(f"Shards: {shard}")
        axis.set_xticks(list(x), chunks)
        axis.set_xlabel("Chunks")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Mean speed (MiB/s) ± SD")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    fig.suptitle("CAREamics whole-image OME-NGFF write benchmark")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    stats: dict[tuple[str, str, str], Stats], rows: list[dict[str, str]]
) -> str:
    """Build a concise interpretation of the benchmark."""
    by_backend: defaultdict[str, list[Stats]] = defaultdict(list)
    by_layout: defaultdict[tuple[str, str], dict[str, Stats]] = defaultdict(dict)
    for (backend, chunks, shards), values in stats.items():
        by_backend[backend].append(values)
        by_layout[(chunks, shards)][backend] = values
    winners: defaultdict[str, int] = defaultdict(int)
    for values in by_layout.values():
        winners[max(values, key=lambda backend: values[backend][0])] += 1

    lines = [
        "CAREamics whole-image OME-NGFF write benchmark summary",
        f"Rows: {len(rows)}; configurations: {len(stats)}",
        "",
    ]
    for backend in sorted(by_backend):
        medians = [value[0] for value in by_backend[backend]]
        deviations = [value[1] for value in by_backend[backend]]
        lines.append(
            f"- {backend}: median {statistics.median(medians):.2f} MiB/s; "
            f"median repeat SD {statistics.median(deviations):.2f} MiB/s; "
            f"best in {winners[backend]}/{len(by_layout)} layouts"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    """Read, summarize, and plot image-write results."""
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    stats = stats_by_configuration(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        stats, rows, args.output_dir / "image_write_mib_per_second.png", args.dpi
    )
    summary = build_summary(stats, rows)
    (args.output_dir / "image_write_summary.txt").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
