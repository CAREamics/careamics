#!/bin/bash
#SBATCH --job-name=switi_inference
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/project/careamics/switi/hpc_outputs/switi_inference/output_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-4

DSET_LIST=(CARE3D_liver CARE3D_zebrafish HT_LIF24_500ms HT_LIF24_5ms PaviaATN)
OVERLAP_LIST=("2 32 32" "2 32 32" "32 32" "32 32" "32 32")

DATASET=${DSET_LIST[$SLURM_ARRAY_TASK_ID]}
OVERLAP=${OVERLAP_LIST[$SLURM_ARRAY_TASK_ID]}

python -u -m scripts.microsplit_inner_tiling_inference \
    --dataset "$DATASET" \
    --split test \
    --overlap $OVERLAP \
    --mmse-count 64 \
    --batch-size 128 \
    --num-workers 3 \
    --device cuda