#!/bin/bash
#SBATCH -p NH100q
#SBATCH -w node07
#SBATCH -J finetune_qwen25_14b_dense_krope
#SBATCH -c 16
#SBATCH -o slimkv-%j.out
#SBATCH -e slimkv-%j.err
#SBATCH --time=UNLIMITED

source "${SLIMKV_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/common.sh"

set -e

echo "Job started on $(hostname) at $(date)"

cd "${PROJECT_ROOT}"

if [[ -n "${SLIMKV_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SLIMKV_CONDA_ENV}"
fi

# Keep the same GPU placement policy as the original training slurm.
export CUDA_VISIBLE_DEVICES=4,5,6,7

NPROC_PER_NODE=4 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MODEL_NAME_OR_PATH=${SLIMKV_OUTPUT_DIR}/slimkv-qwen25-14b-pretrain-dense-krope/checkpoint-3000 \
BEACON_KV_COMPRESSION=none \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_WINDOW=2048 \
BEACON_STRIDE=2048 \
OUTPUT_NAME=slimkv-qwen25-14b-finetune-dense-krope-481632 \
OCCUPY_AFTER_FINETUNE=False \
bash "${PROJECT_ROOT}/scripts/training/finetune_qwen25_14B.sh"

echo "Job finished at $(date)"
