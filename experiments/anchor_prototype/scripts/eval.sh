#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  USER CONFIGURATION
# ================================================================

# --- Model path ---
# Finetuned checkpoint path (a specific checkpoint-* directory).
# A base Hugging Face model can be used with EVAL_MODE=original.
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${SLIMKV_BASE_MODEL_DIR:-../model}/Qwen3-1.7B}"
FINETUNE_DIR="${FINETUNE_DIR:-}"

# --- GPU ---
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# --- Evaluation mode ---
# slimkv: enable SlimKV compression (default).
# original: use the original model with full attention.
EVAL_MODE="${EVAL_MODE:-original}"

# --- Context mode ---
# full: evaluate with the context included.
# close_book: omit context, keeping only the task template and question.
CONTEXT_MODE="${CONTEXT_MODE:-full}"

# --- Data and output ---
# If unset, search local dataset locations or download from Hugging Face.
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-eval}"
RESULT_ROOT="${RESULT_ROOT:-data/results/longbench}"

# --- Evaluation parameters ---
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-hf}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
DTYPE="${DTYPE:-bf16}"

# --- LongBench tasks ---
TASKS=(
  narrativeqa
  qasper
  multifieldqa_en
  hotpotqa
  2wikimqa
  # musique
  # gov_report
  # qmsum
  # multi_news
  # trec
  # triviaqa
  # samsum
  # lcc
  # repobench-p
)

# ================================================================
# The following settings normally need no changes.
# ================================================================

if type module &>/dev/null; then
  if [[ -n "${SLIMKV_CUDA_MODULE-cuda12.8/toolkit/12.8.1}" ]]; then
    module load "${SLIMKV_CUDA_MODULE-cuda12.8/toolkit/12.8.1}"
  fi 2>/dev/null || true
fi

if [[ "${CUDA_HOME:-}" == */bin/nvcc ]] || [[ -f "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$(cd "$(dirname "${CUDA_HOME}")/.." && pwd)"
elif [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
mkdir -p "${TRITON_CACHE_DIR}"
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=0
export MAP_NUM_PROC="${MAP_NUM_PROC:-32}"
export MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-10}"

# --- Locate the checkpoint ---
if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  if compgen -G "${FINETUNE_DIR}/checkpoint-*" > /dev/null 2>&1; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${FINETUNE_DIR}/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${FINETUNE_DIR}" ]]; then
    MODEL_NAME_OR_PATH="${FINETUNE_DIR}"
  else
    echo "ERROR: 找不到模型，请设置 MODEL_NAME_OR_PATH 或 FINETUNE_DIR" >&2
    exit 1
  fi
fi

# If the path is a parent directory without config.json, choose its latest checkpoint.
if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  if compgen -G "${MODEL_NAME_OR_PATH}/checkpoint-*" > /dev/null 2>&1; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${MODEL_NAME_OR_PATH}/checkpoint-* | sort -V | tail -n 1)"
  fi
fi

# --- Build mode arguments ---
mode_suffix=""
mode_args=()
case "${EVAL_MODE}" in
  slimkv)
    mode_suffix="slimkv"
    ;;
  original)
    mode_suffix="original"
    mode_args+=(--no_slimkv)
    ;;
  *)
    echo "ERROR: 不支持的 EVAL_MODE=${EVAL_MODE}，可选: slimkv | original" >&2
    exit 1
    ;;
esac

context_suffix=""
context_args=()
case "${CONTEXT_MODE}" in
  full)
    context_suffix=""
    context_args+=(--context_mode full)
    ;;
  close_book)
    context_suffix="-closebook"
    context_args+=(--context_mode close_book)
    ;;
  *)
    echo "ERROR: 不支持的 CONTEXT_MODE=${CONTEXT_MODE}，可选: full | close_book" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES

data_args=()
if [[ -n "${EVAL_DATA}" ]]; then
  data_args+=(--eval_data "${EVAL_DATA}")
fi

echo "=========================================="
echo " SlimKV LongBench Evaluation"
echo "=========================================="
echo "  MODEL : ${MODEL_NAME_OR_PATH}"
echo "  MODE  : ${EVAL_MODE} (${mode_suffix})"
echo "  CONTEXT: ${CONTEXT_MODE}"
echo "  GPUS  : ${CUDA_VISIBLE_DEVICES}"
echo "  TASKS : ${TASKS[*]}"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.eval_longbench \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  "${data_args[@]}" \
  --output_dir "${RESULT_ROOT}" \
  --result_dir "${RESULT_DIR:-${OUTPUT_NAME}-${mode_suffix}${context_suffix}}" \
  --tasks "${TASKS[@]}" \
  "${mode_args[@]}" \
  "${context_args[@]}" \
  --attn_impl "${ATTN_IMPL}" \
  --chat_template "${CHAT_TEMPLATE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --dtype "${DTYPE}"
