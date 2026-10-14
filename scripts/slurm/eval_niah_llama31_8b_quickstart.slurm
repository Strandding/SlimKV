#!/bin/bash
#SBATCH -J niah_llama31_8b_dense_qs
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

# Pass OPENAI_API_KEY from submit environment.
# Example: OPENAI_API_KEY=sk-... sbatch ${PROJECT_ROOT}/scripts/slurm/eval_niah_llama31_8b_quickstart.slurm
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Please export it before sbatch." >&2
  exit 1
fi

# Quick Start: one length x one depth (fast smoke test)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29563 \
torchrun --nproc_per_node=4 -m main.eval_needle \
  --model_name_or_path ${SLIMKV_OUTPUT_DIR}/slimkv-llama31-8b-finetune-dense/checkpoint-2000 \
  --enable_beacon \
  --beacon_window 1024 \
  --beacon_stride 1024 \
  --beacon_ratio 16 \
  --beacon_ratio_mix sequence \
  --beacon_skip_rope_for_beacon_k False \
  --beacon_shared_kv_down_proj False \
  --attn_impl flash_attention_2 \
  --dtype bf16 \
  --batch_size 1 \
  --haystack_path ${SLIMKV_DATA_ROOT}/needle/PaulGrahamEssays \
  --result_dir llama31_8b_beacon_dense_r16_qs_65k \
  --min_length 4096 \
  --max_length 65536 \
  --num_length_interval 10 \
  --min_depth 0 \
  --max_depth 100 \
  --num_depth_interval 10 \
  --max_new_tokens 50 \
  --gpt_eval

echo "Job finished at $(date)"
