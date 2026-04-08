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

BEACON_K_LATENT_HEAD_DIM="${BEACON_K_LATENT_HEAD_DIM:-64}"
BEACON_V_LATENT_HEAD_DIM="${BEACON_V_LATENT_HEAD_DIM:-64}"
# output_name="${OUTPUT_NAME:-beacon-qwen2-finetune-lowrank-nokrope-shared-down-kv${BEACON_K_LATENT_HEAD_DIM}}"
output_name="${OUTPUT_NAME:-test-${BEACON_K_LATENT_HEAD_DIM}}"
OUTPUT_DIR="data/outputs/${output_name}"
data_root="${DATA_ROOT:-/dataset/common/tzh/.dataset/long-llm}"
default_pretrain_dir="data/outputs/beacon-qwen2-pretrain-lowrank-nokrope-shared-down-kv${BEACON_K_LATENT_HEAD_DIM}-train"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-lowrank_svd}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-True}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  if compgen -G "${default_pretrain_dir}/checkpoint-*" > /dev/null; then
    MODEL_NAME_OR_PATH="$(printf '%s\n' ${default_pretrain_dir}/checkpoint-* | sort -V | tail -n 1)"
  elif [[ -d "${default_pretrain_dir}" ]]; then
    MODEL_NAME_OR_PATH="${default_pretrain_dir}"
  else
    echo "Cannot find pretrained weights under ${default_pretrain_dir}. Set MODEL_NAME_OR_PATH manually." >&2
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
      # With --save_only_model + DeepSpeed, checkpoint-* may contain only model weights
      # and cannot be used by resume_from_checkpoint. Fall back to weight-only restart.
      MODEL_NAME_OR_PATH="${RESUME_FROM_CHECKPOINT}"
      echo "Found model-only checkpoint at ${RESUME_FROM_CHECKPOINT}; restart from model weights (step resets to 0)."
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
    echo "Given checkpoint has no DeepSpeed global_step* state; restart from model weights (step resets to 0)."
  fi
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

export CUDA_VISIBLE_DEVICES=4,5,6,7
# export MODEL_NAME_OR_PATH="/dataset/common/tzh/model/Qwen2-7B-Instruct"

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node "${NPROC_PER_NODE}" $DDP -m main.train \
--output_dir "${OUTPUT_DIR}" \
--model_name_or_path "${MODEL_NAME_OR_PATH}" \
--train_data \
"${data_root}/gpt/one_detail_book.train.16K.json" \
"${data_root}/gpt/one_detail_paper.train.16K.json" \
"${data_root}/longalpaca/train.json" \
"${data_root}/booksum/train.16K.json" \
"${data_root}/needle/train.16K.json" \
"${data_root}/redpajama/train.json[5000]" \
--max_length 20000 \
--min_length 7200 \
--group_by_stride strict \
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
--beacon_skip_rope_for_beacon_k True \
--beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
"${beacon_kv_args[@]}" \
--attn_impl flash_attention_2 \
--learning_rate 1e-5 \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
--gradient_checkpointing \
--use_reentrant False \
--save_only_model \
--num_train_epochs 1 \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit "${SAVE_TOTAL_LIMIT}" \
--logging_dir "${LOG_DIR}" \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json \
--chat_template qwen \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"
