#!/usr/bin/env bash
# Submit one MicroSplit paper script to SLURM (run it from the VDI or hnode02).
#
#   bash submit.sh <dataset> <condition> <mode> [extra script flags]
#
#   datasets/conditions: ht_lif24 {5ms,3split_500ms}, ht_h24 default, ht_t24 default,
#     chicago_sch23 {4ch,ch01,ch23}, ht_p23a default, ht_p23b default,
#     cbg_z18 default, cbg_n18 default, hhmi_d25 default,
#     pavia_p24 {high_highskew,medium_balanced}
#   modes:
#     smoke    every stage starts: fresh chain, chain from the bundled N2V, and a
#              predict of the bundled checkpoint, all on tiny budgets and crops
#     predict  full test-split predict (mmse 50) of the bundled checkpoint
#     e2e      all four stages at the bundled run's settings (resubmit to continue)
#
# Env: PARTITION (default dgx), GRES (default gpu:71gb:1), MEM, HOURS, SUBMIT=1 to send.
# Calls the project venv's python directly: no uv sync (a sync in one job removes
# packages from another job starting on the same venv).
set -euo pipefail
ROOT=/project/careamics/experiments/paper/msplit
PY=/home/igor.zubarev/projects/careamics/.venv/bin/python
[ "$#" -ge 3 ] || { sed -n '2,20p' "$0"; exit 2; }
DS=$1; COND=$2; MODE=$3; shift 3
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # scripts next to this file
SCRIPT=$SRC/$DS/run_$DS.py
[ -f "$SCRIPT" ] || { echo "no script $SCRIPT" >&2; exit 2; }

case "$DS" in   # memory: the whole dataset is loaded as float32
  chicago_sch23) MEM_D=$([ "$COND" = 4ch ] && echo 320G || echo 200G) ;;
  hhmi_d25|cbg_z18|ht_t24) MEM_D=128G ;;
  *) MEM_D=96G ;;
esac
case "$MODE" in
  smoke)   HOURS_D=03:00:00 ;;
  predict) HOURS_D=24:00:00 ;;
  e2e)     HOURS_D=28:15:00 ;;
  *) echo "mode must be smoke, predict or e2e" >&2; exit 2 ;;
esac
PARTITION=${PARTITION:-dgx}; GRES=${GRES:-gpu:71gb:1}
MEM=${MEM:-$MEM_D}; HOURS=${HOURS:-$HOURS_D}

RUN="$PY $SCRIPT --condition $COND"
case "$MODE" in
  smoke)
    CMD="$RUN --smoke --experiment-name smoke $*"
    if [ "$DS" != chicago_sch23 ] && [ "$COND" != 3split_500ms ]; then
      CMD="$CMD; $RUN --smoke --skip-n2v --experiment-name smoke_bundled_n2v $*"
    fi
    CMD="$CMD; $RUN --smoke --skip-n2v --skip-nm --skip-train --experiment-name smoke_bundled_ckpt $*"
    ;;
  predict) CMD="$RUN --skip-n2v --skip-nm --skip-train --experiment-name predict_bundled $*" ;;
  e2e)     CMD="$RUN $*" ;;
esac

TAG=${DS}_${COND}_${MODE}
mkdir -p "$ROOT/logs"
ARGS=(--partition="$PARTITION" --gres="$GRES" --mem="$MEM" --cpus-per-task=8
      --time="$HOURS" --job-name="$TAG" --output="$ROOT/logs/${TAG}_%j.log"
      --wrap="cd $ROOT/$DS; export PYTHONUNBUFFERED=1; $CMD")
echo "+ sbatch ${ARGS[*]}" >&2
if [ "${SUBMIT:-0}" = 1 ]; then
  if command -v sbatch >/dev/null; then sbatch --parsable "${ARGS[@]}"
  else ssh hpc "$(printf '%q ' sbatch --parsable "${ARGS[@]}")"; fi
else
  echo "(dry run; SUBMIT=1 to send)" >&2
fi
