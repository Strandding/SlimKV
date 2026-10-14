#!/bin/bash
#SBATCH -J eval_llama31_8b_raw_nokrope
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

CUDA_VISIBLE_DEVICES=4,6,7 \
NPROC_PER_NODE=3 \
MASTER_PORT=29541 \
MODEL_NAME_OR_PATH=${SLIMKV_BASE_MODEL_DIR}/Llama-3.1-8B-Instruct \
EVAL_MODE=no_beacon \
RAW_NOKROPE=True \
OUTPUT_NAME=llama31-8b-raw-nokrope \
bash "${PROJECT_ROOT}/scripts/evaluation/eval_longbench_llama31_8B.sh"

echo "Job finished at $(date)"
