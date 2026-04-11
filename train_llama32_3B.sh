#!/usr/bin/env bash
# Dense-beacon pretraining script for Llama-3.2-3B-Instruct.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATA_ROOT="${DATA_ROOT:-/dataset/common/tzh/.dataset/long-llm}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/redpajama/train.json}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/dataset/common/tzh/.model/Llama-3.2-3B-Instruct}"
OUTPUT_NAME="${OUTPUT_NAME:-beacon-llama32-3b-pretrain-dense}"
OUTPUT_DIR="data/outputs/${OUTPUT_NAME}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
MASTER_PORT="${MASTER_PORT:-29534}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# ================================

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Training data not found: ${TRAIN_DATA}" >&2
  exit 1
fi

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

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "========== Run Summary =========="
echo "script: train_llama32_3B.sh"
echo "mode: train (dense beacon)"
echo "model_or_ckpt: ${MODEL_NAME_OR_PATH}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "resume_from: ${RESUME_FROM_CHECKPOINT}"
else
  echo "resume_from: none"
fi
echo "train_data: ${TRAIN_DATA}"
echo "output_dir: ${OUTPUT_DIR}"
echo "gpus: ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "beacon: window=1024 stride=1024 ratio='4 8 16 32' param='q k v' pos=interleave"
echo "attn_impl: flash_attention_2 | dtype: bf16"
echo "log_file: ${LOG_FILE}"
echo "================================="

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data "${TRAIN_DATA}" \
--min_length 2400 \
--max_length 20000 \
--group_by_stride strict \
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
--chat_template "${CHAT_TEMPLATE}" \
--attn_impl flash_attention_2 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit "${SAVE_TOTAL_LIMIT}" \
--evaluation_strategy steps \
--num_train_epochs 1 \
--logging_dir "${LOG_DIR}" \
--logging_steps 10 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"

# Training finished — occupy GPUs to hold the cards.
echo "Training complete. Occupying GPUs to hold resources..."
python -m src.cache
