#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J eval_llama31_8b_dense_beaconk_nokrope
#SBATCH -c 16
#SBATCH -o /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_dense_beaconk_nokrope.%j.out
#SBATCH -e /export/home2/zihan/activation_beacon/data/outputs/logs/eval_llama31_8b_dense_beaconk_nokrope.%j.err
#SBATCH --time=UNLIMITED
#SBATCH --dependency=afterany:434188

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd /export/home2/zihan/activation_beacon

source /export/home2/zihan/miniconda3/etc/profile.d/conda.sh
conda activate beacon

CUDA_VISIBLE_DEVICES=4,6,7 \
NPROC_PER_NODE=3 \
MASTER_PORT=29561 \
MODEL_NAME_OR_PATH=/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-dense/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=8 \
BEACON_RATIO_MIX=sequence \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_ENHANCE_K_POS_RESIDUAL=False \
OUTPUT_NAME=beacon-llama31-8b-finetune-dense-beaconk-nokrope \
OCCUPY_AFTER_EVAL=False \
bash /export/home2/zihan/activation_beacon/eval_longbench_llama31_8B.sh

echo "Job finished at $(date)"
