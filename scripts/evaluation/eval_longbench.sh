#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

# ================================================================
#  USER CONFIGURATION  —  只需修改这里
# ================================================================

# --- 模型路径 ---
# 微调后的 checkpoint 路径（指向具体的 checkpoint-* 目录）
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${SLIMKV_OUTPUT_DIR}/slimkv-qwen2-finetune-lowrank-nokrope-shared-down-kv64/checkpoint-3000}"

# --- GPU ---
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

# --- 评测模式 ---
# beacon8: 开启 beacon 压缩（默认）
# no_beacon: 不使用 beacon（全量 attention）
# question_only: 只对 question 部分做 beacon
EVAL_MODE="${EVAL_MODE:-beacon8}"

# --- 数据 & 输出 ---
EVAL_DATA="${EVAL_DATA:-${SLIMKV_DATA_ROOT}/longbench}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-qwen2-finetune-lowrank-kv64}"
RESULT_ROOT="${RESULT_ROOT:-data/results/longbench}"

# --- Beacon 参数（仅 beacon8 / question_only 模式生效）---
BEACON_RATIO="${BEACON_RATIO:-16}"
BEACON_RATIO_MIX="${BEACON_RATIO_MIX:-sequence}"
BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-auto}"
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K:-auto}"
BEACON_ENHANCE_K_POS_RESIDUAL="${BEACON_ENHANCE_K_POS_RESIDUAL:-auto}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-auto}"
BEACON_HEADWISE_UP_PROJ="${BEACON_HEADWISE_UP_PROJ:-auto}"
BEACON_UP_PROJ_GROUP_SIZE="${BEACON_UP_PROJ_GROUP_SIZE:-auto}"

# --- 评测超参 ---
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
DTYPE="${DTYPE:-bf16}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"
EVAL_SAMPLE_SHUFFLE="${EVAL_SAMPLE_SHUFFLE:-True}"
OCCUPY_AFTER_EVAL="${OCCUPY_AFTER_EVAL:-False}"
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

normalize_bool_or_auto() {
  local v="$1"
  case "${v}" in
    True|true) echo "True" ;;
    False|false) echo "False" ;;
    auto|AUTO) echo "auto" ;;
    *)
      return 1
      ;;
  esac
}

read_ckpt_bool_flag() {
  local config_file="$1"
  local key="$2"
  if [[ -z "${config_file}" || ! -f "${config_file}" ]]; then
    echo ""
    return 0
  fi
  local raw
  raw="$(grep -Po "\"${key}\"\\s*:\\s*\\K(true|false)" "${config_file}" | head -n 1 || true)"
  case "${raw}" in
    true) echo "True" ;;
    false) echo "False" ;;
    *) echo "" ;;
  esac
}

normalize_int_or_auto() {
  local v="$1"
  case "${v}" in
    auto|AUTO) echo "auto" ;;
    *)
      if [[ "${v}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${v}"
      else
        return 1
      fi
      ;;
  esac
}

read_ckpt_int_flag() {
  local config_file="$1"
  local key="$2"
  if [[ -z "${config_file}" || ! -f "${config_file}" ]]; then
    echo ""
    return 0
  fi
  local raw
  raw="$(grep -Po "\"${key}\"\\s*:\\s*\\K[0-9]+" "${config_file}" | head -n 1 || true)"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    echo "${raw}"
  else
    echo ""
  fi
}

# 如果指向的是父目录（无 config.json），自动找最新 checkpoint
if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  if compgen -G "${MODEL_NAME_OR_PATH}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${MODEL_NAME_OR_PATH}/checkpoint-* | sort -V | tail -n 1)"
  fi
fi

if [[ -d "${MODEL_NAME_OR_PATH}" ]] && [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  echo "ERROR: ${MODEL_NAME_OR_PATH} 下没有 config.json，请指向具体的 checkpoint 目录" >&2
  exit 1
fi

if ! BEACON_SKIP_ROPE_FOR_BEACON_K="$(normalize_bool_or_auto "${BEACON_SKIP_ROPE_FOR_BEACON_K}")"; then
  echo "ERROR: BEACON_SKIP_ROPE_FOR_BEACON_K must be one of {True, False, auto}" >&2
  exit 1
fi
if ! BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="$(normalize_bool_or_auto "${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}")"; then
  echo "ERROR: BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K must be one of {True, False, auto}" >&2
  exit 1
fi
if ! BEACON_ENHANCE_K_POS_RESIDUAL="$(normalize_bool_or_auto "${BEACON_ENHANCE_K_POS_RESIDUAL}")"; then
  echo "ERROR: BEACON_ENHANCE_K_POS_RESIDUAL must be one of {True, False, auto}" >&2
  exit 1
fi
if ! BEACON_SHARED_KV_DOWN_PROJ="$(normalize_bool_or_auto "${BEACON_SHARED_KV_DOWN_PROJ}")"; then
  echo "ERROR: BEACON_SHARED_KV_DOWN_PROJ must be one of {True, False, auto}" >&2
  exit 1
fi
if ! BEACON_HEADWISE_UP_PROJ="$(normalize_bool_or_auto "${BEACON_HEADWISE_UP_PROJ}")"; then
  echo "ERROR: BEACON_HEADWISE_UP_PROJ must be one of {True, False, auto}" >&2
  exit 1
fi
if ! BEACON_UP_PROJ_GROUP_SIZE="$(normalize_int_or_auto "${BEACON_UP_PROJ_GROUP_SIZE}")"; then
  echo "ERROR: BEACON_UP_PROJ_GROUP_SIZE must be a positive integer or auto" >&2
  exit 1
fi

CONFIG_JSON_PATH=""
if [[ -f "${MODEL_NAME_OR_PATH}/config.json" ]]; then
  CONFIG_JSON_PATH="${MODEL_NAME_OR_PATH}/config.json"
fi

RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K}"
RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}"
RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL="${BEACON_ENHANCE_K_POS_RESIDUAL}"
RESOLVED_BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ}"
RESOLVED_BEACON_HEADWISE_UP_PROJ="${BEACON_HEADWISE_UP_PROJ}"
RESOLVED_BEACON_UP_PROJ_GROUP_SIZE="${BEACON_UP_PROJ_GROUP_SIZE}"
BEACON_SKIP_ROPE_SOURCE="manual"
BEACON_CHUNK_FIXED_ROPE_SOURCE="manual"
BEACON_ENHANCE_SOURCE="manual"
BEACON_SHARED_DOWN_SOURCE="manual"
BEACON_HEADWISE_UP_SOURCE="manual"
BEACON_UP_GROUP_SOURCE="manual"

if [[ "${BEACON_SKIP_ROPE_FOR_BEACON_K}" == "auto" ]]; then
  RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K="$(read_ckpt_bool_flag "${CONFIG_JSON_PATH}" "beacon_skip_rope_for_beacon_k")"
  if [[ -z "${RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K}" ]]; then
    RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K="False"
    BEACON_SKIP_ROPE_SOURCE="default(False,missing_in_ckpt)"
  else
    BEACON_SKIP_ROPE_SOURCE="checkpoint"
  fi
fi

if [[ "${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}" == "auto" ]]; then
  RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="$(read_ckpt_bool_flag "${CONFIG_JSON_PATH}" "beacon_chunk_fixed_rope_for_beacon_k")"
  if [[ -z "${RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}" ]]; then
    RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="False"
    BEACON_CHUNK_FIXED_ROPE_SOURCE="default(False,missing_in_ckpt)"
  else
    BEACON_CHUNK_FIXED_ROPE_SOURCE="checkpoint"
  fi
fi

if [[ "${BEACON_ENHANCE_K_POS_RESIDUAL}" == "auto" ]]; then
  RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL="$(read_ckpt_bool_flag "${CONFIG_JSON_PATH}" "beacon_enhance_k_pos_residual")"
  if [[ -z "${RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL}" ]]; then
    RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL="False"
    BEACON_ENHANCE_SOURCE="default(False,missing_in_ckpt)"
  else
    BEACON_ENHANCE_SOURCE="checkpoint"
  fi
fi

if [[ "${BEACON_SHARED_KV_DOWN_PROJ}" == "auto" ]]; then
  RESOLVED_BEACON_SHARED_KV_DOWN_PROJ="$(read_ckpt_bool_flag "${CONFIG_JSON_PATH}" "beacon_shared_kv_down_proj")"
  if [[ -z "${RESOLVED_BEACON_SHARED_KV_DOWN_PROJ}" ]]; then
    RESOLVED_BEACON_SHARED_KV_DOWN_PROJ="False"
    BEACON_SHARED_DOWN_SOURCE="default(False,missing_in_ckpt)"
  else
    BEACON_SHARED_DOWN_SOURCE="checkpoint"
  fi
fi

if [[ "${BEACON_HEADWISE_UP_PROJ}" == "auto" ]]; then
  RESOLVED_BEACON_HEADWISE_UP_PROJ="$(read_ckpt_bool_flag "${CONFIG_JSON_PATH}" "beacon_headwise_up_proj")"
  if [[ -z "${RESOLVED_BEACON_HEADWISE_UP_PROJ}" ]]; then
    RESOLVED_BEACON_HEADWISE_UP_PROJ="False"
    BEACON_HEADWISE_UP_SOURCE="compat_default(False,missing_in_ckpt)"
  else
    BEACON_HEADWISE_UP_SOURCE="checkpoint"
  fi
fi

if [[ "${BEACON_UP_PROJ_GROUP_SIZE}" == "auto" ]]; then
  RESOLVED_BEACON_UP_PROJ_GROUP_SIZE="$(read_ckpt_int_flag "${CONFIG_JSON_PATH}" "beacon_up_proj_group_size")"
  if [[ -z "${RESOLVED_BEACON_UP_PROJ_GROUP_SIZE}" ]]; then
    RESOLVED_BEACON_UP_PROJ_GROUP_SIZE="1"
    BEACON_UP_GROUP_SOURCE="default(1,missing_in_ckpt)"
  else
    BEACON_UP_GROUP_SOURCE="checkpoint"
  fi
fi

# 自动查找 LongBench 数据路径
if [[ ! -f "${EVAL_DATA}/${TASKS[0]}.jsonl" ]]; then
  for p in \
    ${SLIMKV_DATA_ROOT}/longbench \
    ${SLIMKV_DATA_ROOT}/longbench \
    ${SLIMKV_DATA_ROOT}/longbench; do
    if [[ -f "${p}/${TASKS[0]}.jsonl" ]]; then
      EVAL_DATA="${p}"; break
    fi
  done
  if [[ ! -f "${EVAL_DATA}/${TASKS[0]}.jsonl" ]]; then
    echo "ERROR: 找不到 LongBench 数据，请设置 EVAL_DATA" >&2
    exit 1
  fi
fi

# 构建评测模式参数
mode_suffix=""
mode_args=()
if [[ "${EVAL_MODE}" != "no_beacon" && "${RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL}" == "True" && "${RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K}" != "True" ]]; then
  echo "ERROR: BEACON_ENHANCE_K_POS_RESIDUAL=True requires BEACON_SKIP_ROPE_FOR_BEACON_K=True" >&2
  exit 1
fi
case "${EVAL_MODE}" in
  beacon8)
    mode_suffix="beacon8"
    mode_args+=(
      --enable_beacon
      --beacon_ratio "${BEACON_RATIO}"
      --beacon_ratio_mix "${BEACON_RATIO_MIX}"
      --beacon_skip_rope_for_beacon_k "${RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K}"
      --beacon_chunk_fixed_rope_for_beacon_k "${RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}"
      --beacon_enhance_k_pos_residual "${RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL}"
      --beacon_shared_kv_down_proj "${RESOLVED_BEACON_SHARED_KV_DOWN_PROJ}"
      --beacon_headwise_up_proj "${RESOLVED_BEACON_HEADWISE_UP_PROJ}"
      --beacon_up_proj_group_size "${RESOLVED_BEACON_UP_PROJ_GROUP_SIZE}"
    )
    ;;
  no_beacon)
    mode_suffix="nobeacon"
    ;;
  question_only)
    mode_suffix="qonly"
    mode_args+=(
      --enable_beacon
      --beacon_ratio "${BEACON_RATIO}"
      --beacon_ratio_mix "${BEACON_RATIO_MIX}"
      --beacon_skip_rope_for_beacon_k "${RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K}"
      --beacon_chunk_fixed_rope_for_beacon_k "${RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}"
      --beacon_enhance_k_pos_residual "${RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL}"
      --beacon_shared_kv_down_proj "${RESOLVED_BEACON_SHARED_KV_DOWN_PROJ}"
      --beacon_headwise_up_proj "${RESOLVED_BEACON_HEADWISE_UP_PROJ}"
      --beacon_up_proj_group_size "${RESOLVED_BEACON_UP_PROJ_GROUP_SIZE}"
      --question_only
    )
    ;;
  *)
    echo "ERROR: 不支持的 EVAL_MODE=${EVAL_MODE}，可选: beacon8 | no_beacon | question_only" >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES
IFS=',' read -r -a __visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT=0
for __gpu_id in "${__visible_gpus[@]}"; do
  if [[ -n "${__gpu_id//[[:space:]]/}" ]]; then
    VISIBLE_GPU_COUNT=$((VISIBLE_GPU_COUNT + 1))
  fi
done
if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
  echo "ERROR: NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPU_COUNT} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi
RESULT_DIR_FINAL="${RESULT_DIR:-${OUTPUT_NAME}-${mode_suffix}}"
eval_sample_args=()
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  eval_sample_args+=(--max_samples_per_task "${MAX_SAMPLES_PER_TASK}")
  eval_sample_args+=(--sample_seed "${EVAL_SAMPLE_SEED}")
  eval_sample_args+=(--sample_shuffle "${EVAL_SAMPLE_SHUFFLE}")
fi

occupy_args=()
if [[ -n "${OCCUPY_MEM_GB}" ]]; then
  occupy_args+=(--mem "${OCCUPY_MEM_GB}")
fi
if [[ "${OCCUPY_NO_COMPUTE}" == "True" ]]; then
  occupy_args+=(--no-compute)
fi

OCCUPY_ARMED=0
occupy_on_exit() {
  local exit_code=$?
  if [[ "${OCCUPY_ARMED}" != "1" ]]; then
    return "${exit_code}"
  fi
  if [[ "${OCCUPY_AFTER_EVAL}" == "True" ]]; then
    if [[ "${exit_code}" -eq 0 ]]; then
      echo "Evaluation complete. Occupying GPUs to hold resources..."
    else
      echo "Evaluation exited with code ${exit_code}. Occupying GPUs to hold resources..."
    fi
    python -m tools.gpu_occupy "${occupy_args[@]}" || true
  fi
  return "${exit_code}"
}
trap occupy_on_exit EXIT

echo "=========================================="
echo " LongBench Evaluation"
echo "=========================================="
echo "  MODEL      : ${MODEL_NAME_OR_PATH}"
echo "  MODE       : ${EVAL_MODE} (${mode_suffix})"
echo "  EVAL_DATA  : ${EVAL_DATA}"
echo "  RESULT_DIR : ${RESULT_ROOT}/${RESULT_DIR_FINAL}"
echo "  GPUS       : ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE}, port=${MASTER_PORT})"
if [[ "${EVAL_MODE}" == "no_beacon" ]]; then
  echo "  BEACON     : disabled"
else
  echo "  BEACON     : ratio=${BEACON_RATIO} mix=${BEACON_RATIO_MIX} k_skip_rope=${RESOLVED_BEACON_SKIP_ROPE_FOR_BEACON_K}(${BEACON_SKIP_ROPE_SOURCE}) k_chunk_fixed_rope=${RESOLVED_BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}(${BEACON_CHUNK_FIXED_ROPE_SOURCE}) k_pos_residual=${RESOLVED_BEACON_ENHANCE_K_POS_RESIDUAL}(${BEACON_ENHANCE_SOURCE}) shared_down=${RESOLVED_BEACON_SHARED_KV_DOWN_PROJ}(${BEACON_SHARED_DOWN_SOURCE}) headwise_up=${RESOLVED_BEACON_HEADWISE_UP_PROJ}(${BEACON_HEADWISE_UP_SOURCE}) up_group=${RESOLVED_BEACON_UP_PROJ_GROUP_SIZE}(${BEACON_UP_GROUP_SOURCE})"
fi
echo "  RUNTIME    : attn=${ATTN_IMPL}, dtype=${DTYPE}, batch=${BATCH_SIZE}, max_len=${MAX_LENGTH}"
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  echo "  SAMPLE_CAP : per_task=${MAX_SAMPLES_PER_TASK}, shuffle=${EVAL_SAMPLE_SHUFFLE}, seed=${EVAL_SAMPLE_SEED}"
else
  echo "  SAMPLE_CAP : full dataset"
fi
echo "  TASKS      : ${TASKS[*]}"
echo "=========================================="

OCCUPY_ARMED=1
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
run_status=$?
exit "${run_status}"
