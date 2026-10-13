#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
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
# output_name="${OUTPUT_NAME:-slimkv-qwen2-finetune-lowrank-nokrope-shared-down-kv${BEACON_K_LATENT_HEAD_DIM}}"
output_name="${OUTPUT_NAME:-test-${BEACON_K_LATENT_HEAD_DIM}}"
OUTPUT_DIR="${SLIMKV_OUTPUT_DIR}/${output_name}"
data_root="${DATA_ROOT:-${SLIMKV_DATA_ROOT}}"
default_pretrain_dir="${SLIMKV_OUTPUT_DIR}/slimkv-qwen2-pretrain-lowrank-nokrope-shared-down-kv${BEACON_K_LATENT_HEAD_DIM}-train"
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-lowrank_svd}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-True}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TOTAL_FINETUNE_SAMPLES="${TOTAL_FINETUNE_SAMPLES:-}"
GROUP_BY_STRIDE="${GROUP_BY_STRIDE:-strict}"
OCCUPY_AFTER_FINETUNE="${OCCUPY_AFTER_FINETUNE:-False}"
OCCUPY_MEM_GB="${OCCUPY_MEM_GB:-}"
OCCUPY_NO_COMPUTE="${OCCUPY_NO_COMPUTE:-False}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
# export MODEL_NAME_OR_PATH="${SLIMKV_BASE_MODEL_DIR}/Qwen2-7B-Instruct"

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
echo "script: finetune.sh"
echo "mode: finetune (beacon + optional lowrank kv)"
echo "model_or_ckpt: ${MODEL_NAME_OR_PATH}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "resume_from: ${RESUME_FROM_CHECKPOINT}"
else
  echo "resume_from: none"
fi
echo "data_root: ${data_root}"
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
echo "beacon: window=2048 stride=2048 ratio='4 8' param='q k v' skip_k_rope=True shared_down=${BEACON_SHARED_KV_DOWN_PROJ}"
echo "kv_compression: ${BEACON_KV_COMPRESSION} (k_dim=${BEACON_K_LATENT_HEAD_DIM}, v_dim=${BEACON_V_LATENT_HEAD_DIM})"
echo "attn_impl: flash_attention_2 | dtype: bf16 | lr: 1e-5"
echo "log_file: ${LOG_FILE}"
echo "================================="

OCCUPY_ARMED=1
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
"${train_length_args[@]}" \
--save_strategy steps \
--save_steps "${SAVE_STEPS}" \
--save_total_limit "${SAVE_TOTAL_LIMIT}" \
--logging_dir "${LOG_DIR}" \
--logging_steps 50 \
--bf16 \
--deepspeed config/deepspeed/stage2.json \
--chat_template qwen \
"${resume_args[@]}" 2>&1 | tee "${LOG_FILE}"
run_status=${PIPESTATUS[0]}
exit "${run_status}"
