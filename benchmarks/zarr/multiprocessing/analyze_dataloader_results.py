"""Summarize and plot CAREamics DataLoader multiprocessing benchmarks."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path(__file__).parent / "results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "plots"
Stats = tuple[float, float]
BACKEND_COLORS = {
    "tensorstore": "#1f77b4",  # blue
    "zarrs": "#2ca02c",  # green
    "zarr": "#ff7f0e",  # orange
}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read results and separate successful and failed cases."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}.")
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    if not successful:
        raise ValueError("The benchmark contains no successful cases.")
    return successful, failed


def stats_by_configuration(
    rows: list[dict[str, str]], metric: str
) -> dict[tuple[str, str, str], Stats]:
    """Aggregate repeats by backend, worker count, and context."""
    values: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["backend"],
            row["workers"],
            row["effective_context"],
        )
        values[key].append(float(row[metric]))
    return {
        key: (
            statistics.median(samples),
            statistics.stdev(samples) if len(samples) > 1 else 0.0,
        )
        for key, samples in values.items()
    }


def plot_metric(
    stats: dict[tuple[str, str, str], Stats],
    rows: list[dict[str, str]],
    metric: str,
    output: Path,
    dpi: int,
) -> None:
    """Plot median metric with individual repeat values."""
    backends = sorted({key[0] for key in stats})
    contexts = sorted({key[2] for key in stats})
    colors = {backend: BACKEND_COLORS.get(backend, "#7f7f7f") for backend in backends}
    fig, axes = plt.subplots(
        1,
        len(contexts),
        figsize=(max(6.5, 4.5 * len(contexts)), 4.5),
        sharey=True,
        squeeze=False,
    )
    for axis, context in zip(axes.flat, contexts, strict=True):
        context_workers = sorted({int(key[1]) for key in stats if key[2] == context})
        x = range(len(context_workers))
        for backend in backends:
            entries = [
                stats.get((backend, str(worker), context)) for worker in context_workers
            ]
            if any(entry is None for entry in entries):
                continue
            medians = [entry[0] for entry in entries if entry is not None]
            axis.plot(
                x,
                medians,
                marker="o",
                color=colors[backend],
                label=backend,
            )
            for position, worker in enumerate(context_workers):
                samples = [
                    float(row[metric])
                    for row in rows
                    if row["backend"] == backend
                    and row["workers"] == str(worker)
                    and row["effective_context"] == context
                ]
                axis.scatter(
                    [position] * len(samples),
                    samples,
                    color=colors[backend],
                    alpha=0.45,
                    s=18,
                    zorder=3,
                )
        axis.set_title(f"Context: {context}")
        axis.set_xticks(list(x), [str(worker) for worker in context_workers])
        axis.set_xlabel("DataLoader workers")
        axis.grid(axis="y", alpha=0.25)
    label = {
        "batches_per_second": "Median batches/s",
        "samples_per_second": "Median samples/s",
        "first_batch_seconds": "Median first-batch latency (s)",
    }[metric]
    axes[0, 0].set_ylabel(f"{label}\n(points: individual repeats)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    fig.suptitle(f"CAREamics DataLoader multiprocessing: {label}")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    rows: list[dict[str, str]],
    failed: list[dict[str, str]],
    stats: dict[tuple[str, str, str], Stats],
) -> str:
    """Build an interpretation of throughput and failures."""
    lines = [
        "CAREamics DataLoader multiprocessing benchmark summary",
        f"Successful rows: {len(rows)}; failed/skipped rows: {len(failed)}",
        "",
    ]
    by_backend: defaultdict[str, list[float]] = defaultdict(list)
    for (backend, _, _), (median, _) in stats.items():
        by_backend[backend].append(median)
    for backend in sorted(by_backend):
        lines.append(
            f"- {backend}: median batches/s across configurations "
            f"{statistics.median(by_backend[backend]):.2f}"
        )
    if failed:
        lines.extend(("", "Failures/skips by backend:"))
        counts: defaultdict[str, int] = defaultdict(int)
        examples: defaultdict[str, str] = defaultdict(str)
        for row in failed:
            backend = row.get("backend", "unknown")
            counts[backend] += 1
            examples.setdefault(backend, row.get("error", ""))
        for backend in sorted(counts):
            lines.append(
                f"- {backend}: {counts[backend]}; example: {examples[backend]}"
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
    """Read, summarize, and plot multiprocessing results."""
    args = build_parser().parse_args()
    rows, failed = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("batches_per_second", "samples_per_second", "first_batch_seconds"):
        stats = stats_by_configuration(rows, metric)
        plot_metric(stats, rows, metric, args.output_dir / f"{metric}.png", args.dpi)
    throughput_stats = stats_by_configuration(rows, "batches_per_second")
    summary = build_summary(rows, failed, throughput_stats)
    (args.output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
