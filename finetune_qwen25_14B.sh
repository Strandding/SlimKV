#!/usr/bin/env bash
# Dense-beacon finetuning script for Qwen2.5-14B-Instruct.

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
DEFAULT_PRETRAIN_DIR="data/outputs/beacon-qwen25-14b-pretrain-dense"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
OUTPUT_NAME="${OUTPUT_NAME:-beacon-qwen25-14b-finetune-dense}"
OUTPUT_DIR="data/outputs/${OUTPUT_NAME}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
MASTER_PORT="${MASTER_PORT:-29554}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TOTAL_FINETUNE_SAMPLES="${TOTAL_FINETUNE_SAMPLES:-}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"
# ================================

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

echo "========== Run Summary =========="
echo "script: finetune_qwen25_14B.sh"
echo "mode: finetune (dense beacon)"
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
echo "beacon: window=2048 stride=2048 ratio='4 8' param='q k v' pos=interleave"
echo "attn_impl: flash_attention_2 | dtype: bf16 | lr: 1e-5"
echo "log_file: ${LOG_FILE}"
echo "================================="

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
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 4 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
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
--deepspeed data/deepspeed/stage2.json \
--chat_template "${CHAT_TEMPLATE}" \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"

# Finetuning finished — occupy GPUs to hold the cards.
echo "Finetuning complete. Occupying GPUs to hold resources..."
python -m src.cache
