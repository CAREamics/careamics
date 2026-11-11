"""Summarize and plot whole-plane Zarr benchmark results."""

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
    Path(__file__).parents[1] / "results" / "plane_io" / "plane_io_results.csv"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "results" / "plane_io"
Stats = tuple[float, float]
BACKEND_COLORS = {
    "tensorstore": "#1f77b4",  # blue
    "zarrs": "#2ca02c",  # green
    "zarr": "#ff7f0e",  # orange
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate benchmark rows."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}.")
    invalid = [row for row in rows if row.get("validated") != "True"]
    if invalid:
        raise ValueError(f"Found {len(invalid)} unvalidated benchmark rows.")
    return rows


def stats_by_configuration(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], Stats]:
    """Aggregate repeats into mean throughput and sample standard deviation."""
    values: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["operation"], row["backend"], row["chunks"], row["shards"])
        values[key].append(float(row["mib_per_second"]))
    return {
        key: (
            statistics.mean(samples),
            statistics.stdev(samples) if len(samples) > 1 else 0.0,
        )
        for key, samples in values.items()
    }


def plot_operation(
    stats: dict[tuple[str, str, str, str], Stats],
    rows: list[dict[str, str]],
    operation: str,
    output: Path,
    dpi: int,
) -> None:
    """Plot means with standard-deviation bands by chunk size."""
    shards = sorted({key[3] for key in stats if key[0] == operation})
    backends = sorted({key[1] for key in stats if key[0] == operation})
    chunks = sorted(
        {key[2] for key in stats if key[0] == operation},
        key=lambda value: tuple(int(part) for part in value.split("x")),
    )
    fig, axes = plt.subplots(
        1,
        len(shards),
        figsize=(max(6.5, 4.5 * len(shards)), 4.5),
        sharey=True,
        squeeze=False,
    )
    colors = {backend: BACKEND_COLORS.get(backend, "#7f7f7f") for backend in backends}
    x = range(len(chunks))
    for axis, shard in zip(axes.flat, shards, strict=True):
        for backend in backends:
            entries = [
                stats.get((operation, backend, chunk, shard)) for chunk in chunks
            ]
            if any(entry is None for entry in entries):
                continue
            values = [
                entry[0] if entry is not None else float("nan") for entry in entries
            ]
            deviations = [entry[1] if entry is not None else 0.0 for entry in entries]
            axis.plot(
                x,
                values,
                marker="o",
                color=colors[backend],
                label=backend,
            )
            axis.fill_between(
                x,
                [
                    value - deviation
                    for value, deviation in zip(values, deviations, strict=True)
                ],
                [
                    value + deviation
                    for value, deviation in zip(values, deviations, strict=True)
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
    fig.suptitle(f"Whole-plane {operation} benchmark")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    stats: dict[tuple[str, str, str, str], Stats],
    rows: list[dict[str, str]],
) -> str:
    """Create a concise text interpretation of the benchmark."""
    lines = [
        "Whole-plane Zarr benchmark summary",
        f"Rows: {len(rows)}; configurations: {len(stats)} median values",
        "",
    ]
    for operation in ("read", "write"):
        operation_values = {
            key: value for key, value in stats.items() if key[0] == operation
        }
        if not operation_values:
            continue
        lines.append(f"{operation.title()}:")
        by_backend: defaultdict[str, list[Stats]] = defaultdict(list)
        winners: defaultdict[str, int] = defaultdict(int)
        by_layout: defaultdict[tuple[str, str], dict[str, Stats]] = defaultdict(dict)
        for (_, backend, chunk, shard), values in operation_values.items():
            by_backend[backend].append(values)
            by_layout[(chunk, shard)][backend] = values
        for values in by_layout.values():
            winners[max(values, key=lambda item: values[item][0])] += 1
        for backend in sorted(by_backend):
            medians = [value[0] for value in by_backend[backend]]
            deviations = [value[1] for value in by_backend[backend]]
            lines.append(
                f"- {backend}: median {statistics.median(medians):.2f} MiB/s; "
                f"median repeat SD {statistics.median(deviations):.2f} MiB/s; "
                f"best in {winners[backend]}/{len(by_layout)} layouts"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    """Read, summarize, and plot plane benchmark results."""
    args = build_parser().parse_args()
    rows = read_rows(args.input)
    stats = stats_by_configuration(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for operation in ("read", "write"):
        plot_operation(
            stats,
            rows,
            operation,
            args.output_dir / f"plane_{operation}_mib_per_second.png",
            args.dpi,
        )
    summary = build_summary(stats, rows)
    (args.output_dir / "plane_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
