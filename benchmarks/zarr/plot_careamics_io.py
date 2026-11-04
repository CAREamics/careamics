"""Generate plots for a CAREamics I/O screening manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(command: list[str]) -> None:
    """Run one plotting command."""
    print("Running:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


def main() -> None:
    """Generate patch, tile, and worker plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failed_runs = [run for run in manifest.get("runs", []) if not run.get("success")]
    if failed_runs:
        raise RuntimeError(
            f"Cannot plot an incomplete screening run: {len(failed_runs)} "
            "child benchmark(s) failed."
        )
    output_dir = (args.output_dir or args.manifest.parent / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(ROOT / "io" / "plot_benchmark_results.py"),
            manifest["patch_read"],
            manifest["tile_write"],
            "--output-dir",
            str(output_dir / "io"),
        ]
    )

    worker_plot_dirs: list[str] = []
    for worker_result in manifest["worker_results"]:
        result_path = Path(worker_result)
        if not result_path.is_absolute():
            result_path = args.manifest.parent / result_path
        if not result_path.exists():
            print(f"Skipping missing worker result: {result_path}", file=sys.stderr)
            continue
        with result_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not any(row.get("status") == "ok" for row in rows):
            print(
                f"Skipping worker result with no successful cases: {result_path}",
                file=sys.stderr,
            )
            continue
        worker_output = output_dir / "workers" / result_path.stem
        run(
            [
                sys.executable,
                str(ROOT / "multiprocessing" / "analyze_dataloader_results.py"),
                "--input",
                str(result_path),
                "--output-dir",
                str(worker_output),
            ]
        )
        worker_plot_dirs.append(str(worker_output))

    summary = {
        "io_plots": str(output_dir / "io"),
        "worker_plots": worker_plot_dirs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
