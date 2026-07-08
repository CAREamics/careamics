#!/bin/bash
#SBATCH --job-name=switi_metrics
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_metrics/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-4

# Compute global metrics (PSNR, LPIPS, MS-SSIM, MicroMS3IM, Pearson) on the
# stitched predictions produced by the inference jobs. Each array task handles
# one dataset and scores both tiling variants. Run configuration (data dir,
# tile size, overlap, mmse count, ckpt) is read from each prediction dir's
# `inference_params.json`, so no per-dataset CLI args are needed here. A GPU is
# requested to accelerate LPIPS.

DSET_LIST=(CARE3D_liver CARE3D_zebrafish HT_LIF24_500ms HT_LIF24_5ms PaviaATN)
DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}

RESULTS_ROOT=/project/careamics/switi/results
MMSE_COUNT=64
TILING_LIST=(inner_tiling sw_inner_tiling)

for TILING in "${TILING_LIST[@]}"; do
    PRED_DIR="$RESULTS_ROOT/$DATASET/predictions_MMSE$MMSE_COUNT/$TILING"
    if [[ ! -f "$PRED_DIR/predictions.npz" || ! -f "$PRED_DIR/inference_params.json" ]]; then
        echo "[skip] $PRED_DIR (missing predictions.npz or inference_params.json)"
        continue
    fi
    echo "[run] computing metrics for $PRED_DIR"
    uv run python -u -m scripts.microsplit_compute_metrics \
        --pred-dir "$PRED_DIR" \
        --split test
done
