#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat >&2 <<'EOF'
Submit one MicroSplit paper script to SLURM.

  bash submit.sh <dataset> <condition> <mode> [extra script flags]

  bash submit.sh ht_p23b default predict            # dry run, prints the sbatch command
  SUBMIT=1 bash submit.sh ht_p23b default predict   # submit
  SUBMIT=1 bash submit.sh chicago_sch23 ch23 e2e    # resubmit to continue training

datasets/conditions: ht_lif24 {5ms,3split_500ms}, ht_h24 default, ht_t24 default,
  chicago_sch23 {4ch,ch01,ch23}, ht_p23a default, ht_p23b default,
  cbg_z18 default, cbg_n18 default, hhmi_d25 default,
  pavia_p24 {high_highskew,medium_balanced}
modes: smoke, predict, e2e
env: PARTITION (dgx), GRES (gpu:71gb:1), MEM, HOURS, SUBMIT=1

Outputs: /project/careamics/experiments/paper/msplit/<dataset>/<condition>/runs/<experiment-name>/
  (<experiment-name>: smoke, smoke_bundled_n2v, smoke_bundled_ckpt, predict_bundled or e2e)
Logs: /project/careamics/experiments/paper/msplit/logs/<dataset>_<condition>_<mode>_<jobid>.log
EOF
  exit 2
}
ROOT=/project/careamics/experiments/paper/msplit
PY=/home/igor.zubarev/projects/careamics/.venv/bin/python
[ "$#" -ge 3 ] || usage
DS=$1; COND=$2; MODE=$3; shift 3
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT=$SRC/run_$DS.py
[ -f "$SCRIPT" ] || { echo "no script $SCRIPT" >&2; exit 2; }

case "$DS" in
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
