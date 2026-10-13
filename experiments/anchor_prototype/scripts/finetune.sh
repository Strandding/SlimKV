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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${SCRIPT_DIR}/../config/deepspeed/stage2.json}"

# ========= Config =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DATA_ROOT="${DATA_ROOT:-/dataset/common/tzh/.dataset/long-llm}"
PRETRAIN_DIR="${PRETRAIN_DIR:-outputs/slimkv-qwen2-pretrain}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/slimkv-qwen2-finetune}"
MAX_LENGTH="${MAX_LENGTH:-20000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
# ==========================

# Auto-find latest checkpoint
if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  if compgen -G "${PRETRAIN_DIR}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${PRETRAIN_DIR}/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${PRETRAIN_DIR}" ]]; then
    MODEL_NAME_OR_PATH="${PRETRAIN_DIR}"
  else
    echo "Cannot find pretrained weights under ${PRETRAIN_DIR}." >&2
    exit 1
  fi
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "Cannot find deepspeed config: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data \
"${DATA_ROOT}/gpt/one_detail_book.train.16K.json" \
"${DATA_ROOT}/gpt/one_detail_paper.train.16K.json" \
"${DATA_ROOT}/longalpaca/train.json" \
"${DATA_ROOT}/booksum/train.16K.json" \
"${DATA_ROOT}/needle/train.16K.json" \
"${DATA_ROOT}/redpajama/train.json[5000]" \
--max_length "${MAX_LENGTH}" \
--min_length 7200 \
--group_by_stride strict \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
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
--chat_template qwen \
2>&1 | tee "${LOG_FILE}"
