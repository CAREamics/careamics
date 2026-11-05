#!/usr/bin/env bash
# Run the light-sheet training/prediction benchmark three times per backend.
#
# Usage:
#   examples/ome-ngff/run_lightsheet_backend_benchmark.sh
#   NUM_EPOCHS=1 OUTPUT_DIR=/tmp/lightsheet-smoke \
#     examples/ome-ngff/run_lightsheet_backend_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/lightsheet_benchmark}"
REPEATS="${REPEATS:-3}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BACKENDS="${BACKENDS:-zarr zarrs tensorstore}"

read -r -a BACKEND_ARGS <<< "${BACKENDS}"

mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
exec uv run python "${SCRIPT_DIR}/benchmark_lightsheet_backends.py" \
    --output-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/lightsheet_training_results.csv" \
    --backends "${BACKEND_ARGS[@]}" \
    --repeats "${REPEATS}" \
    --num-epochs "${NUM_EPOCHS}"
