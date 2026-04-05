#!/usr/bin/env bash
set -euo pipefail

# ================================================================
#  USER CONFIGURATION
# ================================================================

# --- 模型路径 ---
# 微调后的 checkpoint 路径（指向具体的 checkpoint-* 目录）
# 也可以指向原始 HF 模型路径（配合 EVAL_MODE=original 使用）
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/zihan/Qwen3-1.7B}"
FINETUNE_DIR="${FINETUNE_DIR:-}"

# --- GPU ---
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# --- 评测模式 ---
# slimkv:   开启 SlimKV 压缩（默认）
# original: 不使用 SlimKV，使用原始模型（全量 attention）
EVAL_MODE="${EVAL_MODE:-original}"

# --- Context 模式 ---
# full:       正常评测（包含 context）
# close_book: 对照组（不提供 context，只保留任务模板与问题）
CONTEXT_MODE="${CONTEXT_MODE:-full}"

# --- 数据 & 输出 ---
# 如果为空，脚本会自动在本地常见路径中搜索或从 HuggingFace 自动下载
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-eval}"
RESULT_ROOT="${RESULT_ROOT:-data/results/longbench}"

# --- 评测超参 ---
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-hf}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
DTYPE="${DTYPE:-bf16}"

# --- LongBench 子任务 ---
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
#  以下内容通常不需要修改
# ================================================================

if type module &>/dev/null; then
  module load cuda12.8/toolkit/12.8.1 2>/dev/null || true
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

# --- 自动查找 checkpoint ---
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

# 如果指向的是父目录（无 config.json），自动找最新 checkpoint
if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  if compgen -G "${MODEL_NAME_OR_PATH}/checkpoint-*" > /dev/null 2>&1; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${MODEL_NAME_OR_PATH}/checkpoint-* | sort -V | tail -n 1)"
  fi
fi

# --- 构建模式参数 ---
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
