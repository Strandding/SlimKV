#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J eval_llama31_8b_dense_beaconk_nokrope
#SBATCH -c 16
#SBATCH -o slimkv-%j.out
#SBATCH -e slimkv-%j.err
#SBATCH --time=UNLIMITED

source "${SLIMKV_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd "${PROJECT_ROOT}"

if [[ -n "${SLIMKV_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SLIMKV_CONDA_ENV}"
fi

CUDA_VISIBLE_DEVICES=4,6,7 \
NPROC_PER_NODE=3 \
MASTER_PORT=29561 \
MODEL_NAME_OR_PATH=${SLIMKV_OUTPUT_DIR}/slimkv-llama31-8b-finetune-dense/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=8 \
BEACON_RATIO_MIX=sequence \
BEACON_SKIP_ROPE_FOR_BEACON_K=True \
BEACON_ENHANCE_K_POS_RESIDUAL=False \
OUTPUT_NAME=slimkv-llama31-8b-finetune-dense-beaconk-nokrope \
OCCUPY_AFTER_EVAL=False \
bash "${PROJECT_ROOT}/scripts/evaluation/eval_longbench_llama31_8B.sh"

echo "Job finished at $(date)"
