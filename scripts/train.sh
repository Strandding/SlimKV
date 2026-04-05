#!/usr/bin/env bash
set -euo pipefail

if type module &>/dev/null; then
  module load cuda12.8/toolkit/12.8.1
fi

if [[ "${CUDA_HOME:-}" == */bin/nvcc ]] || [[ -f "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$(cd "$(dirname "${CUDA_HOME}")/.." && pwd)"
elif [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
mkdir -p "${TRITON_CACHE_DIR}"

export MASTER_PORT=29501
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/../config/deepspeed/stage2.json}"

# ========= Config =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_ROOT="${DATA_ROOT:-/data/long-llm}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/zihan/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/slimkv-qwen3-pretrain}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"  # strict | relaxed | none

# SlimKV options
ANCHOR_KV_TYPE="${ANCHOR_KV_TYPE:-full}"               # full | lowrank
LATENT_DIM="${LATENT_DIM:-64}"                          # only used when lowrank
SKIP_ANCHOR_ROPE_K="${SKIP_ANCHOR_ROPE_K:-False}"       # True | False
SHARED_KV_DOWN="${SHARED_KV_DOWN:-False}"               # True | False (only used when lowrank)
# ==========================

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Cannot find deepspeed config: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

group_by_stride_args=()
if [[ "${GROUP_BY_STRIDE}" != "none" ]]; then
  group_by_stride_args=(--group_by_stride "${GROUP_BY_STRIDE}")
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data "${DATA_ROOT}/redpajama/train.json" \
--min_length 2400 \
--max_length "${MAX_LENGTH}" \
"${group_by_stride_args[@]}" \
--anchor_kv_type "${ANCHOR_KV_TYPE}" \
--latent_dim "${LATENT_DIM}" \
--skip_anchor_rope_k "${SKIP_ANCHOR_ROPE_K}" \
--shared_kv_down "${SHARED_KV_DOWN}" \
--attn_impl flash_attention_2 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing \
--save_only_model \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit 2 \
--num_train_epochs 1 \
--logging_dir "${LOG_DIR}" \
--logging_steps 50 \
--bf16 \
--deepspeed "${DEEPSPEED_CONFIG}" \
2>&1 | tee "${LOG_FILE}"
