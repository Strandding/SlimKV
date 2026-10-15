#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
# Beacon pretraining script for Qwen2.5-14B-Instruct (dense or low-rank).

if type module &>/dev/null; then
  if [[ -n "${SLIMKV_CUDA_MODULE-cuda12.8/toolkit/12.8.1}" ]]; then
    module load "${SLIMKV_CUDA_MODULE-cuda12.8/toolkit/12.8.1}"
  fi
fi

if [[ "${CUDA_HOME:-}" == */bin/nvcc ]] || [[ -f "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$(cd "$(dirname "${CUDA_HOME}")/.." && pwd)"
elif [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
mkdir -p "${TRITON_CACHE_DIR}"

# Reduce PyTorch CUDA allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ========= Configurable parameters =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATA_ROOT="${DATA_ROOT:-${SLIMKV_DATA_ROOT}}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/redpajama/train.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${SLIMKV_BASE_MODEL_DIR}/Qwen2.5-14B-Instruct}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-qwen25-14b-pretrain-dense}"
OUTPUT_DIR="${SLIMKV_OUTPUT_DIR}/${OUTPUT_NAME}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen2.5}"
MASTER_PORT="${MASTER_PORT:-29544}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-none}"
BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-True}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-False}"
BEACON_WINDOW="${BEACON_WINDOW:-2048}"
BEACON_STRIDE="${BEACON_STRIDE:-2048}"
BEACON_KV_LATENT_COMPRESS_RATIO="${BEACON_KV_LATENT_COMPRESS_RATIO:-2}"
BEACON_K_LATENT_HEAD_DIM_LIST="${BEACON_K_LATENT_HEAD_DIM_LIST:-}"
BEACON_V_LATENT_HEAD_DIM_LIST="${BEACON_V_LATENT_HEAD_DIM_LIST:-}"
BEACON_ORIGINAL_HEAD_DIM="${BEACON_ORIGINAL_HEAD_DIM:-}"
TOTAL_TRAIN_SAMPLES="${TOTAL_TRAIN_SAMPLES:-24000}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"
# ================================

infer_head_dim() {
  local model_path="$1"
  if [[ -n "${BEACON_ORIGINAL_HEAD_DIM}" ]]; then
    echo "${BEACON_ORIGINAL_HEAD_DIM}"
    return 0
  fi
  if [[ -f "${model_path}/config.json" ]]; then
    python - "${model_path}/config.json" <<'PY'
import json, sys
with open(sys.argv[1], "r") as f:
    cfg = json.load(f)
head_dim = cfg.get("head_dim")
if head_dim is None:
    hs = cfg.get("hidden_size")
    nh = cfg.get("num_attention_heads")
    if isinstance(hs, int) and isinstance(nh, int) and nh > 0:
        head_dim = hs // nh
if isinstance(head_dim, int) and head_dim > 0:
    print(head_dim)
PY
  fi
}

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Training data not found: ${TRAIN_DATA}" >&2
  exit 1
fi

if [[ "${BEACON_SHARED_KV_DOWN_PROJ}" == "True" && "${BEACON_SKIP_ROPE_FOR_BEACON_K}" != "True" ]]; then
  echo "Invalid config: BEACON_SHARED_KV_DOWN_PROJ=True requires BEACON_SKIP_ROPE_FOR_BEACON_K=True" >&2
  exit 1
fi

beacon_kv_args=()
BEACON_RESOLVED_HEAD_DIM=""
BEACON_LATENT_HEAD_DIM=""
BEACON_K_LATENT_HEAD_DIM_LIST_ARR=()
BEACON_V_LATENT_HEAD_DIM_LIST_ARR=()
case "${BEACON_KV_COMPRESSION}" in
  none)
    ;;
  lowrank_svd)
    if [[ -n "${BEACON_K_LATENT_HEAD_DIM_LIST}" || -n "${BEACON_V_LATENT_HEAD_DIM_LIST}" ]]; then
      if [[ -z "${BEACON_K_LATENT_HEAD_DIM_LIST}" || -z "${BEACON_V_LATENT_HEAD_DIM_LIST}" ]]; then
        echo "Invalid config: set both BEACON_K_LATENT_HEAD_DIM_LIST and BEACON_V_LATENT_HEAD_DIM_LIST, or neither." >&2
        exit 1
      fi
      read -r -a BEACON_K_LATENT_HEAD_DIM_LIST_ARR <<< "${BEACON_K_LATENT_HEAD_DIM_LIST}"
      read -r -a BEACON_V_LATENT_HEAD_DIM_LIST_ARR <<< "${BEACON_V_LATENT_HEAD_DIM_LIST}"
      if (( ${#BEACON_K_LATENT_HEAD_DIM_LIST_ARR[@]} == 0 || ${#BEACON_V_LATENT_HEAD_DIM_LIST_ARR[@]} == 0 )); then
        echo "Invalid BEACON_K_LATENT_HEAD_DIM_LIST / BEACON_V_LATENT_HEAD_DIM_LIST: empty after parsing." >&2
        exit 1
      fi
      if (( ${#BEACON_K_LATENT_HEAD_DIM_LIST_ARR[@]} != ${#BEACON_V_LATENT_HEAD_DIM_LIST_ARR[@]} )); then
        echo "Invalid config: K/V layerwise list lengths differ (${#BEACON_K_LATENT_HEAD_DIM_LIST_ARR[@]} vs ${#BEACON_V_LATENT_HEAD_DIM_LIST_ARR[@]})." >&2
        exit 1
      fi
      for dim in "${BEACON_K_LATENT_HEAD_DIM_LIST_ARR[@]}"; do
        if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
          echo "Invalid per-layer latent dim in BEACON_K_LATENT_HEAD_DIM_LIST: ${dim}" >&2
          exit 1
        fi
      done
      for dim in "${BEACON_V_LATENT_HEAD_DIM_LIST_ARR[@]}"; do
        if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
          echo "Invalid per-layer latent dim in BEACON_V_LATENT_HEAD_DIM_LIST: ${dim}" >&2
          exit 1
        fi
      done
      BEACON_LATENT_HEAD_DIM="variable-kv-separate"
    else
      if ! [[ "${BEACON_KV_LATENT_COMPRESS_RATIO}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid BEACON_KV_LATENT_COMPRESS_RATIO: ${BEACON_KV_LATENT_COMPRESS_RATIO}" >&2
        exit 1
      fi
      BEACON_RESOLVED_HEAD_DIM="$(infer_head_dim "${MODEL_NAME_OR_PATH}")"
      if [[ -z "${BEACON_RESOLVED_HEAD_DIM}" ]]; then
        echo "Cannot infer head_dim from ${MODEL_NAME_OR_PATH}. Set BEACON_ORIGINAL_HEAD_DIM manually." >&2
        exit 1
      fi
      BEACON_LATENT_HEAD_DIM=$(( BEACON_RESOLVED_HEAD_DIM / BEACON_KV_LATENT_COMPRESS_RATIO ))
      if (( BEACON_LATENT_HEAD_DIM < 1 )); then
        echo "Invalid latent dim: head_dim=${BEACON_RESOLVED_HEAD_DIM}, ratio=${BEACON_KV_LATENT_COMPRESS_RATIO}" >&2
        exit 1
      fi
    fi
    beacon_kv_args+=(--beacon_kv_compression "${BEACON_KV_COMPRESSION}")
    if [[ -n "${BEACON_K_LATENT_HEAD_DIM_LIST}" ]]; then
      beacon_kv_args+=(--beacon_k_latent_head_dim_list "${BEACON_K_LATENT_HEAD_DIM_LIST_ARR[@]}")
      beacon_kv_args+=(--beacon_v_latent_head_dim_list "${BEACON_V_LATENT_HEAD_DIM_LIST_ARR[@]}")
    else
      beacon_kv_args+=(--beacon_k_latent_head_dim "${BEACON_LATENT_HEAD_DIM}")
      beacon_kv_args+=(--beacon_v_latent_head_dim "${BEACON_LATENT_HEAD_DIM}")
    fi
    ;;
  *)
    echo "Unsupported BEACON_KV_COMPRESSION: ${BEACON_KV_COMPRESSION}" >&2
    exit 1
    ;;
esac

resume_args=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" || "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
    if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
      RESUME_FROM_CHECKPOINT="$(printf '%s\n' "${OUTPUT_DIR}"/checkpoint-* | sort -V | tail -n 1)"
    else
      echo "No checkpoint found under ${OUTPUT_DIR}, cannot auto-resume." >&2
      exit 1
    fi
  fi
  resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi

train_length_args=()
group_by_stride_args=()
if [[ -n "${GROUP_BY_STRIDE}" && "${GROUP_BY_STRIDE}" != "none" ]]; then
  group_by_stride_args+=(--group_by_stride "${GROUP_BY_STRIDE}")
fi
GLOBAL_BATCH_SIZE=$(( PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS ))
MAX_STEPS=""
if [[ -n "${TOTAL_TRAIN_SAMPLES}" ]]; then
  if ! [[ "${TOTAL_TRAIN_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TOTAL_TRAIN_SAMPLES: ${TOTAL_TRAIN_SAMPLES}" >&2
    exit 1
  fi
  if (( TOTAL_TRAIN_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "TOTAL_TRAIN_SAMPLES (${TOTAL_TRAIN_SAMPLES}) must be divisible by global_batch_size (${GLOBAL_BATCH_SIZE})." >&2
    exit 1
  fi
  MAX_STEPS=$(( TOTAL_TRAIN_SAMPLES / GLOBAL_BATCH_SIZE ))
  if (( MAX_STEPS < 1 )); then
    echo "Computed MAX_STEPS is < 1. Check TOTAL_TRAIN_SAMPLES and batch settings." >&2
    exit 1
  fi
  train_length_args+=(--max_steps "${MAX_STEPS}")
else
  train_length_args+=(--num_train_epochs 1)
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "========== Run Summary =========="
echo "script: train_qwen25_14B.sh"
echo "mode: train (beacon configurable)"
echo "model_or_ckpt: ${MODEL_NAME_OR_PATH}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "resume_from: ${RESUME_FROM_CHECKPOINT}"
else
  echo "resume_from: none"
fi
echo "train_data: ${TRAIN_DATA}"
echo "output_dir: ${OUTPUT_DIR}"
echo "gpus: ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "batch: per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} -> global_batch=${GLOBAL_BATCH_SIZE}"
if [[ -n "${TOTAL_TRAIN_SAMPLES}" ]]; then
  echo "train_budget: total_samples=${TOTAL_TRAIN_SAMPLES} -> max_steps=${MAX_STEPS}"
else
echo "train_budget: total_samples=all (1 epoch)"
fi
echo "group_by_stride: ${GROUP_BY_STRIDE}"
echo "beacon: window=${BEACON_WINDOW} stride=${BEACON_STRIDE} ratio='8' param='q k v' pos=interleave"
echo "beacon_extra: kv_compression=${BEACON_KV_COMPRESSION} k_skip_rope=${BEACON_SKIP_ROPE_FOR_BEACON_K} shared_down=${BEACON_SHARED_KV_DOWN_PROJ}"
if [[ "${BEACON_KV_COMPRESSION}" == "lowrank_svd" ]]; then
  if [[ -n "${BEACON_K_LATENT_HEAD_DIM_LIST}" ]]; then
    echo "beacon_lowrank: per_layer_k_latent_dims='${BEACON_K_LATENT_HEAD_DIM_LIST}'"
    echo "beacon_lowrank: per_layer_v_latent_dims='${BEACON_V_LATENT_HEAD_DIM_LIST}'"
  else
    echo "beacon_lowrank: head_dim=${BEACON_RESOLVED_HEAD_DIM} ratio=${BEACON_KV_LATENT_COMPRESS_RATIO} -> latent_dim=${BEACON_LATENT_HEAD_DIM}"
  fi
else
  echo "beacon_lowrank: disabled"
fi
echo "attn_impl: flash_attention_2 | dtype: bf16"
echo "log_file: ${LOG_FILE}"
echo "================================="

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.training.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data "${TRAIN_DATA}" \
--min_length 2400 \
--max_length 20000 \
"${group_by_stride_args[@]}" \
--enable_beacon \
--beacon_window "${BEACON_WINDOW}" \
--beacon_stride "${BEACON_STRIDE}" \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 4 8 16 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--beacon_skip_rope_for_beacon_k "${BEACON_SKIP_ROPE_FOR_BEACON_K}" \
--beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
"${beacon_kv_args[@]}" \
--chat_template "${CHAT_TEMPLATE}" \
--attn_impl flash_attention_2 \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit "${SAVE_TOTAL_LIMIT}" \
--evaluation_strategy steps \
"${train_length_args[@]}" \
--logging_dir "${LOG_DIR}" \
--logging_steps 10 \
--bf16 \
--deepspeed config/deepspeed/stage2.json \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"
run_status=${PIPESTATUS[0]}
exit "${run_status}"
