#!/usr/bin/env bash
# Run the 20-epoch light-sheet training/prediction benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec uv run python "${SCRIPT_DIR}/training/benchmark_train_predict.py" "$@"
