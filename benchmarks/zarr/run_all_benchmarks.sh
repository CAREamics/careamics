#!/usr/bin/env bash
# Run the complete CAREamics OME-NGFF/Zarr benchmark suite and generate plots.
#
# Usage:
#   benchmarks/zarr/run_all_benchmarks.sh
#   OUTPUT_DIR=/tmp/zarr-benchmark REPEATS=3 benchmarks/zarr/run_all_benchmarks.sh
#
# The full suite is intentionally sizeable. Override the environment variables
# below for a smoke run or for a platform-specific benchmark configuration.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/${RUN_ID}}"
REPEATS="${REPEATS:-3}"
WARMUPS="${WARMUPS:-1}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
ARRAY_SHAPE="${ARRAY_SHAPE:-2048,2048}"
PATCH_SIZES="${PATCH_SIZES:-64x64 128x128 256x256}"
TILE_SIZES="${TILE_SIZES:-128x128 256x256}"
CHUNKS="${CHUNKS:-64x64 128x128 256x256}"
SHARDS="${SHARDS:-none 256x256 512x512}"
PLANE_CHUNKS="${PLANE_CHUNKS:-64,64 128,128 256,256}"
PLANE_SHARDS="${PLANE_SHARDS:-none 256,256 512,512}"
BACKENDS="${BACKENDS:-zarr zarrs tensorstore}"
WORKERS="${WORKERS:-0 2 4}"

read -r -a PATCH_SIZE_ARGS <<< "${PATCH_SIZES}"
read -r -a TILE_SIZE_ARGS <<< "${TILE_SIZES}"
read -r -a CHUNK_ARGS <<< "${CHUNKS}"
read -r -a SHARD_ARGS <<< "${SHARDS}"
read -r -a PLANE_CHUNK_ARGS <<< "${PLANE_CHUNKS}"
read -r -a PLANE_SHARD_ARGS <<< "${PLANE_SHARDS}"
read -r -a BACKEND_ARGS <<< "${BACKENDS}"
read -r -a WORKER_ARGS <<< "${WORKERS}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/plots" "${OUTPUT_DIR}/logs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

run_logged() {
    local name="$1"
    shift
    echo "==> ${name}"
    echo "    $*"
    "$@" 2>&1 | tee "${OUTPUT_DIR}/logs/${name}.log"
}

run_logged plane \
    uv run python "${SCRIPT_DIR}/io/benchmark_plane.py" \
    --array-shape "${ARRAY_SHAPE}" \
    --chunks "${PLANE_CHUNK_ARGS[@]}" \
    --shards "${PLANE_SHARD_ARGS[@]}" \
    --backends "${BACKEND_ARGS[@]}" \
    --repeats "${REPEATS}" \
    --warmups "${WARMUPS}" \
    --output "${OUTPUT_DIR}/plane_results.csv"

run_logged image_write \
    uv run python "${SCRIPT_DIR}/io/benchmark_write_image.py" \
    --array-shape "${ARRAY_SHAPE}" \
    --chunks "${PLANE_CHUNK_ARGS[@]}" \
    --shards "${PLANE_SHARD_ARGS[@]}" \
    --backends "${BACKEND_ARGS[@]}" \
    --repeats "${REPEATS}" \
    --warmups "${WARMUPS}" \
    --output "${OUTPUT_DIR}/image_write_results.csv"

run_logged careamics_io \
    uv run python "${SCRIPT_DIR}/benchmark_careamics_io.py" \
    --output-dir "${OUTPUT_DIR}/careamics_io" \
    --array-shape "${ARRAY_SHAPE}" \
    --patch-sizes "${PATCH_SIZE_ARGS[@]}" \
    --tile-sizes "${TILE_SIZE_ARGS[@]}" \
    --chunks "${CHUNK_ARGS[@]}" \
    --shards "${SHARD_ARGS[@]}" \
    --backends "${BACKEND_ARGS[@]}" \
    --workers "${WORKER_ARGS[@]}" \
    --repeats "${REPEATS}" \
    --warmups "${WARMUPS}" \
    --measure-batches "${MEASURE_BATCHES}"

run_logged plane_analysis \
    uv run python "${SCRIPT_DIR}/io/analyze_plane_results.py" \
    --input "${OUTPUT_DIR}/plane_results.csv" \
    --output-dir "${OUTPUT_DIR}/plots/plane"

run_logged image_write_analysis \
    uv run python "${SCRIPT_DIR}/io/analyze_image_write_results.py" \
    --input "${OUTPUT_DIR}/image_write_results.csv" \
    --output-dir "${OUTPUT_DIR}/plots/image_write"

run_logged careamics_io_analysis \
    uv run python "${SCRIPT_DIR}/plot_careamics_io.py" \
    "${OUTPUT_DIR}/careamics_io/manifest.json" \
    --output-dir "${OUTPUT_DIR}/plots/careamics_io"

echo
echo "Benchmark suite complete. Results: ${OUTPUT_DIR}"
