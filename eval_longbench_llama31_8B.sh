#!/usr/bin/env bash

# ================================================================
#  USER CONFIGURATION  —  只需修改这里
# ================================================================

# --- 模型路径 ---
# 若不手动指定 MODEL_NAME_OR_PATH，则自动在 DEFAULT_FINETUNE_DIR 下选择最新 checkpoint-*
DEFAULT_FINETUNE_DIR="${DEFAULT_FINETUNE_DIR:-data/outputs/beacon-llama31-8b-finetune-dense}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"

# --- GPU ---
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29531}"

# --- 评测模式 ---
# beacon8: 开启 beacon 压缩（默认）
# no_beacon: 不使用 beacon（全量 attention）
# question_only: 只对 question 部分做 beacon
EVAL_MODE="${EVAL_MODE:-beacon8}"

# --- 数据 & 输出 ---
EVAL_DATA="${EVAL_DATA:-/dataset/common/tzh/dataset/long-llm/longbench}"
OUTPUT_NAME="${OUTPUT_NAME:-beacon-llama31-8b-finetune-dense}"
RESULT_ROOT="${RESULT_ROOT:-data/results/longbench}"

# --- Beacon 参数（仅 beacon8 / question_only 模式生效）---
BEACON_RATIO="${BEACON_RATIO:-8}"
BEACON_RATIO_MIX="${BEACON_RATIO_MIX:-sequence}"

# --- 评测超参 ---
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
DTYPE="${DTYPE:-bf16}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"
EVAL_SAMPLE_SHUFFLE="${EVAL_SAMPLE_SHUFFLE:-True}"
OCCUPY_AFTER_EVAL="${OCCUPY_AFTER_EVAL:-True}"
OCCUPY_MEM_GB="${OCCUPY_MEM_GB:-}"
OCCUPY_NO_COMPUTE="${OCCUPY_NO_COMPUTE:-False}"

# --- LongBench 子任务（默认去掉中文子集） ---
# 已移除: multifieldqa_zh, dureader, vcsum, lsht, passage_retrieval_zh
TASKS=(
  narrativeqa
  qasper
  multifieldqa_en
  hotpotqa
  2wikimqa
  musique
  gov_report
  qmsum
  multi_news
  trec
  triviaqa
  samsum
  passage_count
  passage_retrieval_en
  lcc
  repobench-p
)

# ================================================================
#  以下内容通常不需要修改
# ================================================================

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
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=0
export MAP_NUM_PROC="${MAP_NUM_PROC:-32}"
export MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-10}"

# 自动解析模型路径：优先用显式 MODEL_NAME_OR_PATH，否则从默认 finetune 目录选最新 ckpt
if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  if compgen -G "${DEFAULT_FINETUNE_DIR}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' "${DEFAULT_FINETUNE_DIR}"/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -f "${DEFAULT_FINETUNE_DIR}/config.json" ]]; then
    MODEL_NAME_OR_PATH="${DEFAULT_FINETUNE_DIR}"
  else
    echo "ERROR: Cannot find checkpoint under ${DEFAULT_FINETUNE_DIR}" >&2
    exit 1
  fi
fi

# 如果传入的是父目录（无 config.json），自动找最新 checkpoint
if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  if compgen -G "${MODEL_NAME_OR_PATH}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' "${MODEL_NAME_OR_PATH}"/checkpoint-* | sort -V | tail -n 1)"
  fi
fi

if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  echo "ERROR: ${MODEL_NAME_OR_PATH} 下没有 config.json，请指向具体 checkpoint 目录" >&2
  exit 1
fi

# 自动查找 LongBench 数据路径
PROBE_TASK="${TASKS[0]}"
if [[ "${PROBE_TASK}" == "all" ]]; then
  PROBE_TASK="narrativeqa"
fi
if [[ ! -f "${EVAL_DATA}/${PROBE_TASK}.jsonl" ]]; then
  for p in \
    /dataset/common/tzh/dataset/long-llm/longbench \
    /dataset/common/tzh/datasets/long-llm/longbench \
    /dataset/common/tzh/.dataset/long-llm/longbench; do
    if [[ -f "${p}/${PROBE_TASK}.jsonl" ]]; then
      EVAL_DATA="${p}"; break
    fi
  done
  if [[ ! -f "${EVAL_DATA}/${PROBE_TASK}.jsonl" ]]; then
    echo "ERROR: 找不到 LongBench 数据，请设置 EVAL_DATA" >&2
    exit 1
  fi
fi

# 构建评测模式参数
mode_suffix=""
mode_args=()
case "${EVAL_MODE}" in
  beacon8)
    mode_suffix="beacon8"
    mode_args+=(--enable_beacon --beacon_ratio "${BEACON_RATIO}" --beacon_ratio_mix "${BEACON_RATIO_MIX}")
    ;;
  no_beacon)
    mode_suffix="nobeacon"
    ;;
  question_only)
    mode_suffix="qonly"
    mode_args+=(--enable_beacon --beacon_ratio "${BEACON_RATIO}" --beacon_ratio_mix "${BEACON_RATIO_MIX}" --question_only)
    ;;
  *)
    echo "ERROR: 不支持的 EVAL_MODE=${EVAL_MODE}，可选: beacon8 | no_beacon | question_only" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES
RESULT_DIR_FINAL="${RESULT_DIR:-${OUTPUT_NAME}-${mode_suffix}}"
eval_sample_args=()
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  eval_sample_args+=(--max_samples_per_task "${MAX_SAMPLES_PER_TASK}")
  eval_sample_args+=(--sample_seed "${EVAL_SAMPLE_SEED}")
  eval_sample_args+=(--sample_shuffle "${EVAL_SAMPLE_SHUFFLE}")
fi

echo "=========================================="
echo " LongBench Evaluation (Llama3.1-8B)"
echo "=========================================="
echo "  MODEL      : ${MODEL_NAME_OR_PATH}"
echo "  MODE       : ${EVAL_MODE} (${mode_suffix})"
echo "  EVAL_DATA  : ${EVAL_DATA}"
echo "  RESULT_DIR : ${RESULT_ROOT}/${RESULT_DIR_FINAL}"
echo "  GPUS       : ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE}, port=${MASTER_PORT})"
if [[ "${EVAL_MODE}" == "no_beacon" ]]; then
  echo "  BEACON     : disabled"
else
  echo "  BEACON     : ratio=${BEACON_RATIO} mix=${BEACON_RATIO_MIX}"
fi
echo "  RUNTIME    : attn=${ATTN_IMPL}, dtype=${DTYPE}, batch=${BATCH_SIZE}, max_len=${MAX_LENGTH}"
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  echo "  SAMPLE_CAP : per_task=${MAX_SAMPLES_PER_TASK}, shuffle=${EVAL_SAMPLE_SHUFFLE}, seed=${EVAL_SAMPLE_SEED}"
else
  echo "  SAMPLE_CAP : full dataset"
fi
echo "  TASKS      : ${TASKS[*]}"
echo "=========================================="

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" $DDP -m main.eval_longbench \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --eval_data "${EVAL_DATA}" \
  --output_dir "${RESULT_ROOT}" \
  --result_dir "${RESULT_DIR_FINAL}" \
  --tasks "${TASKS[@]}" \
  "${mode_args[@]}" \
  --attn_impl "${ATTN_IMPL}" \
  --chat_template "${CHAT_TEMPLATE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --dtype "${DTYPE}" \
  "${eval_sample_args[@]}"

occupy_args=()
if [[ -n "${OCCUPY_MEM_GB}" ]]; then
  occupy_args+=(--mem "${OCCUPY_MEM_GB}")
fi
if [[ "${OCCUPY_NO_COMPUTE}" == "True" ]]; then
  occupy_args+=(--no-compute)
fi

if [[ "${OCCUPY_AFTER_EVAL}" == "True" ]]; then
  echo "Evaluation complete. Occupying GPUs to hold resources..."
  python -m src.cache "${occupy_args[@]}"
fi
