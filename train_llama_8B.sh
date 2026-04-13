#!/usr/bin/env bash
# Dense-beacon pretraining script for Llama-3.1-8B-Instruct.

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

# Reduce PyTorch CUDA allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ========= 主要可调参数 =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATA_ROOT="${DATA_ROOT:-/dataset/common/tzh/.dataset/long-llm}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/redpajama/train.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/dataset/common/tzh/.model/Llama-3.1-8B-Instruct}"
OUTPUT_NAME_BASE="${OUTPUT_NAME_BASE:-beacon-llama31-8b-pretrain}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
MASTER_PORT="${MASTER_PORT:-29514}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
BEACON_WINDOW="${BEACON_WINDOW:-1024}"
BEACON_STRIDE="${BEACON_STRIDE:-1024}"
BEACON_RATIO="${BEACON_RATIO:-4 8 16}"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-none}"
BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-False}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-False}"
BEACON_KV_LATENT_COMPRESS_RATIO="${BEACON_KV_LATENT_COMPRESS_RATIO:-2}"
BEACON_ORIGINAL_HEAD_DIM="${BEACON_ORIGINAL_HEAD_DIM:-}"
TOTAL_TRAIN_SAMPLES="${TOTAL_TRAIN_SAMPLES:-48000}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"
OCCUPY_AFTER_TRAIN="${OCCUPY_AFTER_TRAIN:-True}"
OCCUPY_MEM_GB="${OCCUPY_MEM_GB:-}"
OCCUPY_NO_COMPUTE="${OCCUPY_NO_COMPUTE:-False}"
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
case "${BEACON_KV_COMPRESSION}" in
  none)
    ;;
  lowrank_svd)
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
    beacon_kv_args+=(--beacon_kv_compression "${BEACON_KV_COMPRESSION}")
    beacon_kv_args+=(--beacon_k_latent_head_dim "${BEACON_LATENT_HEAD_DIM}")
    beacon_kv_args+=(--beacon_v_latent_head_dim "${BEACON_LATENT_HEAD_DIM}")
    ;;
  *)
    echo "Unsupported BEACON_KV_COMPRESSION: ${BEACON_KV_COMPRESSION}" >&2
    exit 1
    ;;
esac

mode_tags=()
if [[ "${BEACON_KV_COMPRESSION}" == "lowrank_svd" ]]; then
  mode_tags+=("lowrank-r${BEACON_KV_LATENT_COMPRESS_RATIO}")
else
  mode_tags+=("dense")
fi
if [[ "${BEACON_SKIP_ROPE_FOR_BEACON_K}" == "True" ]]; then
  mode_tags+=("nokropek")
fi
if [[ "${BEACON_SHARED_KV_DOWN_PROJ}" == "True" ]]; then
  mode_tags+=("shareddown")
fi
MODE_SUFFIX="$(IFS=-; echo "${mode_tags[*]}")"
if [[ -z "${OUTPUT_NAME}" ]]; then
  OUTPUT_NAME="${OUTPUT_NAME_BASE}-${MODE_SUFFIX}"
fi
OUTPUT_DIR="data/outputs/${OUTPUT_NAME}"

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
echo "script: train_llama_8B.sh"
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
echo "occupy_after_train: ${OCCUPY_AFTER_TRAIN}"
echo "beacon: window=${BEACON_WINDOW} stride=${BEACON_STRIDE} ratio='${BEACON_RATIO}' param='q k v' pos=interleave"
echo "beacon_extra: kv_compression=${BEACON_KV_COMPRESSION} k_skip_rope=${BEACON_SKIP_ROPE_FOR_BEACON_K} shared_down=${BEACON_SHARED_KV_DOWN_PROJ}"
if [[ "${BEACON_KV_COMPRESSION}" == "lowrank_svd" ]]; then
  echo "beacon_lowrank: head_dim=${BEACON_RESOLVED_HEAD_DIM} ratio=${BEACON_KV_LATENT_COMPRESS_RATIO} -> latent_dim=${BEACON_LATENT_HEAD_DIM}"
else
  echo "beacon_lowrank: disabled"
fi
echo "attn_impl: flash_attention_2 | dtype: bf16"
echo "log_file: ${LOG_FILE}"
echo "================================="

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.train \
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
--beacon_sink_size 1 \
--beacon_ratio ${BEACON_RATIO} \
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
--deepspeed data/deepspeed/stage2.json \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"

occupy_args=()
if [[ -n "${OCCUPY_MEM_GB}" ]]; then
  occupy_args+=(--mem "${OCCUPY_MEM_GB}")
fi
if [[ "${OCCUPY_NO_COMPUTE}" == "True" ]]; then
  occupy_args+=(--no-compute)
fi

if [[ "${OCCUPY_AFTER_TRAIN}" == "True" ]]; then
  echo "Training complete. Occupying GPUs to hold resources..."
  python -m src.cache "${occupy_args[@]}"
fi
