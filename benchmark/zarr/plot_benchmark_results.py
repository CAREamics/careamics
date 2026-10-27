"""Plot and summarize CAREamics Zarr benchmark results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("mib_per_second", "patches_per_second", "tiles_per_second")
LINE_STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS = (
    RESULTS_DIR / "read_results.csv",
    RESULTS_DIR / "write_results.csv",
)


def read_results(path: Path) -> list[dict[str, str]]:
    """Read benchmark results from a CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Benchmark CSV path.

    Returns
    -------
    list[dict[str, str]]
        Parsed result rows.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Benchmark results are empty: {path}.")
    return rows


def benchmark_kind(rows: Sequence[Mapping[str, str]]) -> str:
    """Determine the benchmark type from its columns.

    Parameters
    ----------
    rows : sequence of mappings
        Benchmark rows.

    Returns
    -------
    str
        Either ``read`` or ``write``.
    """
    columns = rows[0].keys()
    if "patch_size" in columns:
        return "read"
    if "tile_size" in columns and "tile_overlap" in columns:
        return "write"
    raise ValueError("Results are neither a read nor a write benchmark.")


def select_metric(rows: Sequence[Mapping[str, str]], requested: str | None) -> str:
    """Select an available throughput metric.

    Parameters
    ----------
    rows : sequence of mappings
        Benchmark rows.
    requested : str or None
        Explicit metric, or ``None`` to select the operation rate.

    Returns
    -------
    str
        Selected metric column.
    """
    available = rows[0].keys()
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Metric {requested!r} is not present in the results.")
        return requested
    for metric in ("mib_per_second", "patches_per_second", "tiles_per_second"):
        if metric in available:
            return metric
    raise ValueError("Results contain no supported throughput metric.")


def aggregate_results(
    rows: Sequence[Mapping[str, str]], metric: str, kind: str
) -> dict[tuple[str, ...], float]:
    """Aggregate repeated measurements using their median.

    Parameters
    ----------
    rows : sequence of mappings
        Benchmark rows.
    metric : str
        Throughput column.
    kind : str
        Benchmark type.

    Returns
    -------
    dict[tuple[str, ...], float]
        Median throughput keyed by configuration and backend.
    """
    fields = configuration_fields(kind)
    values: defaultdict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in (*fields, "backend"))
        values[key].append(float(row[metric]))
    return {key: statistics.median(samples) for key, samples in values.items()}


def configuration_fields(kind: str) -> tuple[str, ...]:
    """Return identifying configuration fields.

    Parameters
    ----------
    kind : str
        Benchmark type.

    Returns
    -------
    tuple[str, ...]
        Configuration column names.
    """
    if kind == "read":
        return "array_shape", "chunks", "shards", "patch_size"
    return "array_shape", "chunks", "shards", "tile_size", "tile_overlap"


def axis_label(row: Mapping[str, str], kind: str) -> str:
    """Build the categorical x-axis label for a result row.

    Parameters
    ----------
    row : mapping
        Benchmark row.
    kind : str
        Benchmark type.

    Returns
    -------
    str
        Patch or tile layout label.
    """
    if kind == "read":
        return row["patch_size"]
    return f'{row["tile_size"]}\noverlap {row["tile_overlap"]}'


def metric_label(metric: str) -> str:
    """Return a display label for a metric.

    Parameters
    ----------
    metric : str
        Metric column.

    Returns
    -------
    str
        Human-readable label.
    """
    labels = {
        "mib_per_second": "Throughput (MiB/s)",
        "patches_per_second": "Patches/s",
        "tiles_per_second": "Tiles/s",
    }
    return labels[metric]


def plot_results(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    kind: str,
    output: Path,
    dpi: int,
) -> None:
    """Plot median throughput by backend and storage layout.

    Parameters
    ----------
    rows : sequence of mappings
        Benchmark rows.
    metric : str
        Throughput column.
    kind : str
        Benchmark type.
    output : pathlib.Path
        Figure destination.
    dpi : int
        Output resolution.

    Returns
    -------
    None
        Figure is written to disk.
    """
    aggregated = aggregate_results(rows, metric, kind)
    shards = list(dict.fromkeys(row["shards"] for row in rows))
    backends = list(dict.fromkeys(row["backend"] for row in rows))
    chunks = list(dict.fromkeys(row["chunks"] for row in rows))
    x_labels = list(dict.fromkeys(axis_label(row, kind) for row in rows))
    x_positions = range(len(x_labels))

    ncols = min(3, len(shards))
    nrows = math.ceil(len(shards) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7, 4.8 * ncols), 4.2 * nrows),
        sharey=True,
        squeeze=False,
    )
    colors = dict(zip(backends, plt.get_cmap("tab10").colors, strict=False))
    styles = dict(zip(chunks, cycle(LINE_STYLES), strict=False))
    markers = dict(zip(chunks, cycle(MARKERS), strict=False))

    fields = configuration_fields(kind)
    array_shapes = list(dict.fromkeys(row["array_shape"] for row in rows))
    if len(array_shapes) != 1:
        raise ValueError("Plot one array shape per CSV file.")

    lookup: dict[tuple[str, str, str, str], float] = {}
    for key, value in aggregated.items():
        config = dict(zip((*fields, "backend"), key, strict=True))
        label = (
            config["patch_size"]
            if kind == "read"
            else f'{config["tile_size"]}\noverlap {config["tile_overlap"]}'
        )
        lookup[(config["shards"], config["backend"], config["chunks"], label)] = value

    for shard, axis in zip(shards, axes.flat, strict=False):
        for backend in backends:
            for chunk in chunks:
                values = [
                    lookup.get((shard, backend, chunk, label)) for label in x_labels
                ]
                if all(value is None for value in values):
                    continue
                axis.plot(
                    x_positions,
                    values,
                    color=colors[backend],
                    linestyle=styles[chunk],
                    marker=markers[chunk],
                    label=f"{backend}, chunks {chunk}",
                )
        axis.set_title(f"Shards: {shard}")
        axis.set_xticks(list(x_positions), x_labels)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Patch size" if kind == "read" else "Tile size / overlap")

    for axis in list(axes.flat)[len(shards) :]:
        axis.set_visible(False)
    axes[0, 0].set_ylabel(metric_label(metric))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=min(3, len(labels)))
    fig.suptitle(
        f"Zarr {kind} benchmark, array shape {array_shapes[0]}", fontsize="x-large"
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def geometric_mean(values: Iterable[float]) -> float:
    """Calculate the geometric mean of positive values.

    Parameters
    ----------
    values : iterable of float
        Positive values.

    Returns
    -------
    float
        Geometric mean.
    """
    samples = list(values)
    return math.exp(sum(math.log(value) for value in samples) / len(samples))


def summarize_results(
    rows: Sequence[Mapping[str, str]], metric: str, kind: str, baseline: str
) -> str:
    """Interpret backend performance across comparable configurations.

    Parameters
    ----------
    rows : sequence of mappings
        Benchmark rows.
    metric : str
        Throughput column.
    kind : str
        Benchmark type.
    baseline : str
        Backend used for relative speedups.

    Returns
    -------
    str
        Plain-text benchmark summary.
    """
    aggregated = aggregate_results(rows, metric, kind)
    fields = configuration_fields(kind)
    by_config: defaultdict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for key, value in aggregated.items():
        by_config[key[:-1]][key[-1]] = value

    backends = list(dict.fromkeys(row["backend"] for row in rows))
    wins = dict.fromkeys(backends, 0)
    for measurements in by_config.values():
        winner = max(measurements, key=measurements.__getitem__)
        wins[winner] += 1

    lines = [
        f"Benchmark: Zarr {kind}",
        f"Metric: {metric_label(metric)}",
        f"Configurations: {len(by_config)}",
        "",
        "Backend summary:",
    ]
    for backend in backends:
        samples = [
            values[backend] for values in by_config.values() if backend in values
        ]
        best_key, best_value = max(
            (
                (config, values[backend])
                for config, values in by_config.items()
                if backend in values
            ),
            key=lambda item: item[1],
        )
        details = ", ".join(
            f"{field}={value}" for field, value in zip(fields, best_key, strict=True)
        )
        lines.append(
            f"- {backend}: median {statistics.median(samples):.3f}; "
            f"fastest in {wins[backend]}/{len(by_config)} configurations; "
            f"best {best_value:.3f} ({details})."
        )

    if baseline not in backends:
        lines.extend(
            ("", f"Baseline {baseline!r} is not present; no speedups reported.")
        )
        return "\n".join(lines) + "\n"

    lines.extend(("", f"Relative to {baseline}:"))
    for backend in backends:
        if backend == baseline:
            continue
        ratios = [
            values[backend] / values[baseline]
            for values in by_config.values()
            if backend in values and baseline in values and values[baseline] > 0
        ]
        if not ratios:
            lines.append(f"- {backend}: no comparable configurations.")
            continue
        faster = sum(ratio > 1 for ratio in ratios)
        lines.append(
            f"- {backend}: {geometric_mean(ratios):.3f}x geometric-mean speedup; "
            f"faster in {faster}/{len(ratios)} comparable configurations."
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="*", default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_plots"))
    parser.add_argument("--metric", choices=METRICS)
    parser.add_argument("--baseline", default="zarr")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> None:
    """Plot and summarize each result file.

    Returns
    -------
    None
        Plots and summaries are written to disk.
    """
    args = build_parser().parse_args()
    for path in args.results:
        rows = read_results(path)
        kind = benchmark_kind(rows)
        metric = select_metric(rows, args.metric)
        stem = f"{path.stem}_{metric}"
        figure_path = args.output_dir / f"{stem}.{args.format}"
        summary_path = args.output_dir / f"{stem}_summary.txt"
        plot_results(rows, metric, kind, figure_path, args.dpi)
        summary_path.write_text(
            summarize_results(rows, metric, kind, args.baseline), encoding="utf-8"
        )
        print(f"Wrote {figure_path} and {summary_path}.")


if __name__ == "__main__":
    main()
