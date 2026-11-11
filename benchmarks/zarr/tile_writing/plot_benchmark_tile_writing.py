"""Plot tile-writing results with the shared patch/tile plotting implementation."""

from __future__ import annotations

import sys
from pathlib import Path

PLOTTER_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PLOTTER_DIR))

from _plot_patch_tile import main as plot_main  # noqa: E402

if __name__ == "__main__":
    default_results = (
        Path(__file__).parents[1]
        / "result"
        / "tile_writing"
        / "tile_writing_results.csv"
    )
    default_output = Path(__file__).parents[1] / "result" / "tile_writing"
    if len(sys.argv) == 1:
        sys.argv.extend([str(default_results), "--output-dir", str(default_output)])
    plot_main()
