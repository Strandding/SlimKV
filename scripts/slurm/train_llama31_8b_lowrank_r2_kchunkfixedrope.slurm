#!/bin/bash
#SBATCH -J train_llama31_8b_r2_kchunkfixed
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
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
MODEL_NAME_OR_PATH=${SLIMKV_OUTPUT_DIR}/slimkv-llama31-8b-lowrank-r2-nokropek \
BEACON_KV_COMPRESSION=lowrank_svd \
BEACON_KV_LATENT_COMPRESS_RATIO=2 \
BEACON_SKIP_ROPE_FOR_BEACON_K=False \
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K=True \
BEACON_ENHANCE_K_POS_RESIDUAL=False \
BEACON_SHARED_KV_DOWN_PROJ=False \
BEACON_HEADWISE_UP_PROJ=False \
BEACON_UP_PROJ_GROUP_SIZE=1 \
TOTAL_TRAIN_SAMPLES=24000 \
OUTPUT_NAME=slimkv-llama31-8b-pretrain-lowrank-r2-kchunkfixedrope \
bash "${PROJECT_ROOT}/scripts/training/train_llama_8B.sh"

echo "Job finished at $(date)"
