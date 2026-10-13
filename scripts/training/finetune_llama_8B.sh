#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
# Dense-beacon finetuning script for Llama-3.1-8B-Instruct.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATA_ROOT="${DATA_ROOT:-${SLIMKV_DATA_ROOT}}"
DEFAULT_PRETRAIN_DIR="${SLIMKV_OUTPUT_DIR}/slimkv-llama31-8b-pretrain-dense"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-llama31-8b-finetune-dense}"
OUTPUT_DIR="${SLIMKV_OUTPUT_DIR}/${OUTPUT_NAME}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
MASTER_PORT="${MASTER_PORT:-29524}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-none}"
BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-False}"
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K:-False}"
BEACON_ENHANCE_K_POS_RESIDUAL="${BEACON_ENHANCE_K_POS_RESIDUAL:-False}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-False}"
BEACON_HEADWISE_UP_PROJ="${BEACON_HEADWISE_UP_PROJ:-False}"
BEACON_UP_PROJ_GROUP_SIZE="${BEACON_UP_PROJ_GROUP_SIZE:-1}"
BEACON_KV_LATENT_COMPRESS_RATIO="${BEACON_KV_LATENT_COMPRESS_RATIO:-2}"
BEACON_KV_LATENT_HEAD_DIM_LIST="${BEACON_KV_LATENT_HEAD_DIM_LIST:-}"
BEACON_K_LATENT_HEAD_DIM_LIST="${BEACON_K_LATENT_HEAD_DIM_LIST:-}"
BEACON_V_LATENT_HEAD_DIM_LIST="${BEACON_V_LATENT_HEAD_DIM_LIST:-}"
BEACON_ORIGINAL_HEAD_DIM="${BEACON_ORIGINAL_HEAD_DIM:-}"
TOTAL_FINETUNE_SAMPLES="${TOTAL_FINETUNE_SAMPLES:-16000}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"
OCCUPY_AFTER_FINETUNE="${OCCUPY_AFTER_FINETUNE:-False}"
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

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  if compgen -G "${DEFAULT_PRETRAIN_DIR}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' "${DEFAULT_PRETRAIN_DIR}"/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${DEFAULT_PRETRAIN_DIR}" ]]; then
    MODEL_NAME_OR_PATH="${DEFAULT_PRETRAIN_DIR}"
  else
    echo "Cannot find pretrained weights under ${DEFAULT_PRETRAIN_DIR}. Set MODEL_NAME_OR_PATH manually." >&2
    exit 1
  fi
fi

if [[ "${BEACON_SHARED_KV_DOWN_PROJ}" == "True" && "${BEACON_SKIP_ROPE_FOR_BEACON_K}" != "True" ]]; then
  echo "Invalid config: BEACON_SHARED_KV_DOWN_PROJ=True requires BEACON_SKIP_ROPE_FOR_BEACON_K=True" >&2
  exit 1
fi
if [[ "${BEACON_ENHANCE_K_POS_RESIDUAL}" == "True" && "${BEACON_SKIP_ROPE_FOR_BEACON_K}" != "True" ]]; then
  echo "Invalid config: BEACON_ENHANCE_K_POS_RESIDUAL=True requires BEACON_SKIP_ROPE_FOR_BEACON_K=True" >&2
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
    if [[ -n "${BEACON_KV_LATENT_HEAD_DIM_LIST}" ]]; then
      echo "Invalid config: BEACON_KV_LATENT_HEAD_DIM_LIST is deprecated. Use BEACON_K_LATENT_HEAD_DIM_LIST and BEACON_V_LATENT_HEAD_DIM_LIST." >&2
      exit 1
    fi
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
if [[ "${RESUME_FROM_CHECKPOINT}" == "auto" || "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    RESUME_FROM_CHECKPOINT="$(printf '%s\n' "${OUTPUT_DIR}"/checkpoint-* | sort -V | tail -n 1)"
    if compgen -G "${RESUME_FROM_CHECKPOINT}/global_step*" > /dev/null; then
      resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
      echo "Resume from checkpoint (DeepSpeed state): ${RESUME_FROM_CHECKPOINT}"
    else
      MODEL_NAME_OR_PATH="${RESUME_FROM_CHECKPOINT}"
      echo "Found model-only checkpoint at ${RESUME_FROM_CHECKPOINT}; restart from model weights."
    fi
  else
    echo "No checkpoint found under ${OUTPUT_DIR}; start a new finetune run."
  fi
elif [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if compgen -G "${RESUME_FROM_CHECKPOINT}/global_step*" > /dev/null; then
    resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    echo "Resume from checkpoint (DeepSpeed state): ${RESUME_FROM_CHECKPOINT}"
  else
    MODEL_NAME_OR_PATH="${RESUME_FROM_CHECKPOINT}"
    echo "Given checkpoint has no DeepSpeed global_step* state; restart from model weights."
  fi
fi

train_length_args=()
group_by_stride_args=()
if [[ -n "${GROUP_BY_STRIDE}" && "${GROUP_BY_STRIDE}" != "none" ]]; then
  group_by_stride_args+=(--group_by_stride "${GROUP_BY_STRIDE}")
fi
GLOBAL_BATCH_SIZE=$(( PER_DEVICE_TRAIN_BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS ))
MAX_STEPS=""
if [[ -n "${TOTAL_FINETUNE_SAMPLES}" ]]; then
  if ! [[ "${TOTAL_FINETUNE_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid TOTAL_FINETUNE_SAMPLES: ${TOTAL_FINETUNE_SAMPLES}" >&2
    exit 1
  fi
  if (( TOTAL_FINETUNE_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "TOTAL_FINETUNE_SAMPLES (${TOTAL_FINETUNE_SAMPLES}) must be divisible by global_batch_size (${GLOBAL_BATCH_SIZE})." >&2
    exit 1
  fi
  MAX_STEPS=$(( TOTAL_FINETUNE_SAMPLES / GLOBAL_BATCH_SIZE ))
  if (( MAX_STEPS < 1 )); then
    echo "Computed MAX_STEPS is < 1. Check TOTAL_FINETUNE_SAMPLES and batch settings." >&2
    exit 1
  fi
  train_length_args+=(--max_steps "${MAX_STEPS}")
else
  train_length_args+=(--num_train_epochs 1)
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

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
  if [[ "${OCCUPY_AFTER_FINETUNE}" == "True" ]]; then
    if [[ "${exit_code}" -eq 0 ]]; then
      echo "Finetuning complete. Occupying GPUs to hold resources..."
    else
      echo "Finetuning exited with code ${exit_code}. Occupying GPUs to hold resources..."
    fi
    python -m tools.gpu_occupy "${occupy_args[@]}" || true
  fi
  return "${exit_code}"
}
trap occupy_on_exit EXIT

echo "========== Run Summary =========="
echo "script: finetune_llama_8B.sh"
echo "mode: finetune (beacon configurable)"
echo "model_or_ckpt: ${MODEL_NAME_OR_PATH}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "resume_from: ${RESUME_FROM_CHECKPOINT}"
else
  echo "resume_from: none"
fi
echo "data_root: ${DATA_ROOT}"
echo "output_dir: ${OUTPUT_DIR}"
echo "gpus: ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "batch: per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} -> global_batch=${GLOBAL_BATCH_SIZE}"
if [[ -n "${TOTAL_FINETUNE_SAMPLES}" ]]; then
  echo "finetune_budget: total_samples=${TOTAL_FINETUNE_SAMPLES} -> max_steps=${MAX_STEPS}"
else
  echo "finetune_budget: total_samples=all (1 epoch)"
fi
echo "group_by_stride: ${GROUP_BY_STRIDE}"
echo "occupy_after_finetune: ${OCCUPY_AFTER_FINETUNE}"
echo "beacon: window=1024 stride=1024 ratio='2 4 8' param='q k v' pos=interleave"
echo "beacon_extra: kv_compression=${BEACON_KV_COMPRESSION} k_skip_rope=${BEACON_SKIP_ROPE_FOR_BEACON_K} k_chunk_fixed_rope=${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K} k_pos_residual=${BEACON_ENHANCE_K_POS_RESIDUAL} shared_down=${BEACON_SHARED_KV_DOWN_PROJ} headwise_up=${BEACON_HEADWISE_UP_PROJ} up_group=${BEACON_UP_PROJ_GROUP_SIZE}"
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
echo "attn_impl: flash_attention_2 | dtype: bf16 | lr: 1e-5"
echo "log_file: ${LOG_FILE}"
echo "================================="

OCCUPY_ARMED=1
torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data \
"${DATA_ROOT}/gpt/one_detail_book.train.16K.json" \
"${DATA_ROOT}/gpt/one_detail_paper.train.16K.json" \
"${DATA_ROOT}/longalpaca/train.json" \
"${DATA_ROOT}/booksum/train.16K.json" \
"${DATA_ROOT}/needle/train.16K.json" \
"${DATA_ROOT}/redpajama/train.json[5000]" \
--max_length 20000 \
--min_length 7200 \
"${group_by_stride_args[@]}" \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 4 8 16 32 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--beacon_skip_rope_for_beacon_k "${BEACON_SKIP_ROPE_FOR_BEACON_K}" \
--beacon_chunk_fixed_rope_for_beacon_k "${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}" \
--beacon_enhance_k_pos_residual "${BEACON_ENHANCE_K_POS_RESIDUAL}" \
--beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
--beacon_headwise_up_proj "${BEACON_HEADWISE_UP_PROJ}" \
--beacon_up_proj_group_size "${BEACON_UP_PROJ_GROUP_SIZE}" \
"${beacon_kv_args[@]}" \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
"${train_length_args[@]}" \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit "${SAVE_TOTAL_LIMIT}" \
--logging_dir "${LOG_DIR}" \
--logging_steps 10 \
--bf16 \
--deepspeed config/deepspeed/stage2.json \
--chat_template "${CHAT_TEMPLATE}" \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"
run_status=${PIPESTATUS[0]}
exit "${run_status}"
