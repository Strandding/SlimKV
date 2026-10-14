#!/bin/bash
#SBATCH -J train_qwen25_14b_dense_krope
#SBATCH -c 16
#SBATCH -o slimkv-%j.out
#SBATCH -e slimkv-%j.err
#SBATCH --time=UNLIMITED

# Select partition and GPU resources with sbatch options.
source "${SLIMKV_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

set -e

echo "Job started on $(hostname) at $(date)"

cd "${PROJECT_ROOT}"

if [[ -n "${SLIMKV_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SLIMKV_CONDA_ENV}"
fi

# Fixed GPU ids requested by user.
export CUDA_VISIBLE_DEVICES=4,5,6,7

NPROC_PER_NODE=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MODEL_NAME_OR_PATH=${SLIMKV_BASE_MODEL_DIR}/Qwen2.5-14B-Instruct \
BEACON_KV_COMPRESSION=none \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_WINDOW=2048 \
BEACON_STRIDE=2048 \
OUTPUT_NAME=slimkv-qwen25-14b-pretrain-dense-krope \
bash "${PROJECT_ROOT}/scripts/training/train_qwen25_14B.sh"

echo "Job finished at $(date)"
