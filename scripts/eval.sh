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

# ========= Config =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EVAL_DATA="${EVAL_DATA:-/dataset/common/tzh/.dataset/long-llm/longbench}"
FINETUNE_DIR="${FINETUNE_DIR:-outputs/slimkv-qwen2-finetune}"
RESULT_DIR="${RESULT_DIR:-outputs/results/longbench}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
# ==========================

# Auto-find latest checkpoint
if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  if compgen -G "${FINETUNE_DIR}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${FINETUNE_DIR}/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${FINETUNE_DIR}" ]]; then
    MODEL_NAME_OR_PATH="${FINETUNE_DIR}"
  else
    echo "Cannot find finetuned weights under ${FINETUNE_DIR}." >&2
    exit 1
  fi
fi

echo "[eval] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"

# Reuse activation_beacon's eval script
cd ../activation_beacon

torchrun --nproc_per_node "${NPROC_PER_NODE}" -m main.eval_longbench \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--eval_data "${EVAL_DATA}" \
--output_dir "${RESULT_DIR}" \
--enable_beacon \
--beacon_ratio 8 \
--beacon_ratio_mix sequence \
--attn_impl flash_attention_2 \
--chat_template qwen \
--batch_size 1 \
--max_length "${MAX_LENGTH}" \
--dtype bf16
