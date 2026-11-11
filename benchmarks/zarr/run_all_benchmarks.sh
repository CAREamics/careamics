#!/usr/bin/env bash
# Run every non-training Zarr benchmark with its built-in defaults.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

uv run python "${SCRIPT_DIR}/plane_io/benchmark_plane_io.py"
uv run python "${SCRIPT_DIR}/plane_io/plot_benchmark_plane_io.py"

uv run python "${SCRIPT_DIR}/prediction_writer/benchmark_plane_write.py"
uv run python "${SCRIPT_DIR}/prediction_writer/plot_benchmark_plane_write.py"

uv run python "${SCRIPT_DIR}/patch_reading/benchmark_patch_reading.py"
uv run python "${SCRIPT_DIR}/patch_reading/plot_benchmark_patch_reading.py"

uv run python "${SCRIPT_DIR}/tile_writing/benchmark_tile_writing.py"
uv run python "${SCRIPT_DIR}/tile_writing/plot_benchmark_tile_writing.py"

uv run python "${SCRIPT_DIR}/data_loader/benchmark_data_loader.py"
uv run python "${SCRIPT_DIR}/data_loader/plot_benchmark_data_loader.py"
