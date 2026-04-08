#!/usr/bin/env bash
# DeepSpeed 导入时会检查 CUDA_HOME/nvcc；计算节点上需加载与 PyTorch 匹配的 CUDA toolkit（本机为 cu128）。
if type module &>/dev/null; then
  module load cuda12.8/toolkit/12.8.1
  # 可选：module load cudnn9.8-cuda12.8/9.8.0.87
fi

# 某些环境会把 CUDA_HOME 设成 nvcc 可执行文件路径，DeepSpeed 会错误拼接为 ".../bin/nvcc/bin/nvcc"。
if [[ "${CUDA_HOME:-}" == */bin/nvcc ]] || [[ -f "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$(cd "$(dirname "${CUDA_HOME}")/.." && pwd)"
elif [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

# 避免 DeepSpeed Triton autotune 缓存落在 NFS。
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_autotune}"
mkdir -p "${TRITON_CACHE_DIR}"

# ========= 主要可调参数，集中放这里 =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATA_ROOT="${DATA_ROOT:-/dataset/common/tzh/.dataset/long-llm}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/redpajama/train.json}"
BEACON_K_LATENT_HEAD_DIM="${BEACON_K_LATENT_HEAD_DIM:-64}"
BEACON_V_LATENT_HEAD_DIM="${BEACON_V_LATENT_HEAD_DIM:-64}"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-lowrank_svd}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-True}"
default_lowrank_dir="data/outputs/beacon-qwen2-pretrain-lowrank-kv${BEACON_K_LATENT_HEAD_DIM}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${default_lowrank_dir}}"
output_name="${OUTPUT_NAME:-beacon-qwen2-pretrain-lowrank-nokrope-shared-down-kv${BEACON_K_LATENT_HEAD_DIM}-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
OUTPUT_DIR="data/outputs/${output_name}"
# ==========================================

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "Training data not found: ${TRAIN_DATA}" >&2
  echo "Set TRAIN_DATA or DATA_ROOT to a valid dataset path before running train.sh." >&2
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

beacon_kv_args=()
case "${BEACON_KV_COMPRESSION}" in
  none)
    ;;
  lowrank_svd)
    beacon_kv_args+=(--beacon_kv_compression "${BEACON_KV_COMPRESSION}")
    beacon_kv_args+=(--beacon_k_latent_head_dim "${BEACON_K_LATENT_HEAD_DIM}")
    beacon_kv_args+=(--beacon_v_latent_head_dim "${BEACON_V_LATENT_HEAD_DIM}")
    ;;
  *)
    echo "Unsupported BEACON_KV_COMPRESSION: ${BEACON_KV_COMPRESSION}" >&2
    exit 1
    ;;
esac

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
export MASTER_PORT=29502

torchrun --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data "${TRAIN_DATA}" \
--min_length 2400 \
--max_length 20000 \
--group_by_stride strict \
--enable_beacon \
--beacon_window 2048 \
--beacon_stride 2048 \
--beacon_attn full-coverage \
--beacon_attend_prev True \
--beacon_sink_size 0 \
--beacon_ratio 8 \
--beacon_ratio_mix step-random \
--beacon_param q k v \
--beacon_pos interleave \
--beacon_skip_rope_for_beacon_k True \
--beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
"${beacon_kv_args[@]}" \
--beacon_loss_mode weighted_after_beacon \
--beacon_loss_weight 1.0 \
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
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"
