"""Plot patch-reading results."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from _plot_patch_tile import main as plot_main

if __name__ == "__main__":
    plot_main()
