#!/bin/bash
#SBATCH -J eval_qwen25_14b_dense_krope
#SBATCH -c 16
#SBATCH -o slimkv-%j.out
#SBATCH -e slimkv-%j.err
#SBATCH --time=UNLIMITED

# Select partition and GPU resources with sbatch options.
source "${SLIMKV_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

cd "${PROJECT_ROOT}"

if [[ -n "${SLIMKV_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SLIMKV_CONDA_ENV}"
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
MASTER_PORT=29552 \
MODEL_NAME_OR_PATH=${SLIMKV_OUTPUT_DIR}/slimkv-qwen25-14b-finetune-dense-krope-481632/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=16 \
BEACON_RATIO_MIX=sequence \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
CHAT_TEMPLATE=qwen \
ATTN_IMPL=flash_attention_2 \
DTYPE=bf16 \
BATCH_SIZE=1 \
MAX_LENGTH=31500 \
OUTPUT_NAME=slimkv-qwen25-14b-tokenr16-dense-krope \
bash "${PROJECT_ROOT}/scripts/evaluation/eval_longbench.sh"

echo "Job finished at $(date)"
