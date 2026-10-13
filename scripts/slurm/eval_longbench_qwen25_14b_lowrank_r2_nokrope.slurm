#!/bin/bash
#SBATCH -p HPCAIq
#SBATCH -w node14
#SBATCH -J eval_qwen25_14b_lowrank_r2_nokrope
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
MASTER_PORT=29551 \
MODEL_NAME_OR_PATH=${SLIMKV_OUTPUT_DIR}/slimkv-qwen25-14b-finetune-lowrank-r2-nokrope/checkpoint-2000 \
EVAL_MODE=beacon8 \
BEACON_RATIO=8 \
BEACON_RATIO_MIX=sequence \
CHAT_TEMPLATE=qwen \
ATTN_IMPL=flash_attention_2 \
DTYPE=bf16 \
BATCH_SIZE=1 \
MAX_LENGTH=31500 \
OUTPUT_NAME=slimkv-qwen25-14b-finetune-lowrank-r2-nokrope \
OCCUPY_AFTER_EVAL=False \
bash "${PROJECT_ROOT}/scripts/evaluation/eval_longbench.sh"

echo "Job finished at $(date)"
