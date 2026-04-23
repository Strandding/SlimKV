#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default: regular LongBench eval path.
PROBE_HEATMAP="${PROBE_HEATMAP:-True}"

# Shared args
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/export/home2/zihan/activation_beacon/data/outputs/beacon-llama31-8b-finetune-dense/checkpoint-2000}"
EVAL_DATA="${EVAL_DATA:-/dataset/common/tzh/.dataset/long-llm/longbench}"
OUTPUT_DIR="${OUTPUT_DIR:-data/results/longbench_probe}"
RESULT_DIR="${RESULT_DIR:-beacon_chunk_heatmap}"
TASKS="${TASKS:-narrativeqa}"
MAX_LENGTH="${MAX_LENGTH:-31500}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
MAP_NUM_PROC="${MAP_NUM_PROC:-32}"
MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-10}"

# Probe-only args
FIGURE_DIR="${FIGURE_DIR:-data/fig}"
FIGURE_NAME="${FIGURE_NAME:-beacon_chunk_attn_heatmap}"
PROBE_CHUNK_IDX="${PROBE_CHUNK_IDX:-5}"
HEATMAP_VMAX="${HEATMAP_VMAX:-0.01}"
PROBE_MAX_NEW_TOKENS="${PROBE_MAX_NEW_TOKENS:-1}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-10}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_SHUFFLE="${SAMPLE_SHUFFLE:-True}"
STOP_AFTER_FIRST_HEATMAP="${STOP_AFTER_FIRST_HEATMAP:-True}"

# Fixed probe setting: 8x compression
BEACON_RATIO="${BEACON_RATIO:-8}"
BEACON_RATIO_MIX="${BEACON_RATIO_MIX:-sequence}"

# Beacon toggles (kept explicit so probe path is reproducible)
BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-False}"
BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K="${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K:-False}"
BEACON_ENHANCE_K_POS_RESIDUAL="${BEACON_ENHANCE_K_POS_RESIDUAL:-False}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-False}"
BEACON_HEADWISE_UP_PROJ="${BEACON_HEADWISE_UP_PROJ:-False}"
BEACON_UP_PROJ_GROUP_SIZE="${BEACON_UP_PROJ_GROUP_SIZE:-1}"

# Low-rank knobs (only needed when checkpoint config does not already carry them)
BEACON_KV_COMPRESSION="${BEACON_KV_COMPRESSION:-}"
BEACON_K_LATENT_HEAD_DIM="${BEACON_K_LATENT_HEAD_DIM:-}"
BEACON_V_LATENT_HEAD_DIM="${BEACON_V_LATENT_HEAD_DIM:-}"

normalize_bool() {
  local v="$1"
  case "${v}" in
    True|true) echo "True" ;;
    False|false) echo "False" ;;
    *)
      echo "ERROR: invalid bool value '${v}'." >&2
      exit 1
      ;;
  esac
}

PROBE_HEATMAP="$(normalize_bool "${PROBE_HEATMAP}")"
SAMPLE_SHUFFLE="$(normalize_bool "${SAMPLE_SHUFFLE}")"
STOP_AFTER_FIRST_HEATMAP="$(normalize_bool "${STOP_AFTER_FIRST_HEATMAP}")"

if [[ "${PROBE_HEATMAP}" == "False" ]]; then
  # Keep the original eval behavior untouched.
  exec bash /export/home2/zihan/activation_beacon/eval_longbench_llama31_8B.sh
fi

if [[ "${ATTN_IMPL}" != "eager" ]]; then
  echo "ERROR: PROBE_HEATMAP=True requires ATTN_IMPL=eager." >&2
  exit 1
fi
if [[ "${BEACON_RATIO}" != "8" ]]; then
  echo "ERROR: PROBE_HEATMAP=True is fixed to BEACON_RATIO=8." >&2
  exit 1
fi
if ! [[ "${PROBE_CHUNK_IDX}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PROBE_CHUNK_IDX must be a positive integer, got '${PROBE_CHUNK_IDX}'." >&2
  exit 1
fi

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "ERROR: set MODEL_NAME_OR_PATH for probe run." >&2
  exit 1
fi

IFS=' ' read -r -a TASK_ARRAY <<< "${TASKS}"
if [[ ${#TASK_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: TASKS is empty." >&2
  exit 1
fi

TASK_ARGS=()
for task in "${TASK_ARRAY[@]}"; do
  TASK_ARGS+=(--tasks "${task}")
done

export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=0
export MAP_NUM_PROC
export MAP_BATCH_SIZE

# Probe path is intentionally single-process to avoid duplicated figures.
MASTER_PORT="${MASTER_PORT:-29571}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export CUDA_VISIBLE_DEVICES

EXTRA_COMPRESSION_ARGS=()
if [[ -n "${BEACON_KV_COMPRESSION}" ]]; then
  EXTRA_COMPRESSION_ARGS+=(--beacon_kv_compression "${BEACON_KV_COMPRESSION}")
fi
if [[ -n "${BEACON_K_LATENT_HEAD_DIM}" ]]; then
  EXTRA_COMPRESSION_ARGS+=(--beacon_k_latent_head_dim "${BEACON_K_LATENT_HEAD_DIM}")
fi
if [[ -n "${BEACON_V_LATENT_HEAD_DIM}" ]]; then
  EXTRA_COMPRESSION_ARGS+=(--beacon_v_latent_head_dim "${BEACON_V_LATENT_HEAD_DIM}")
fi

torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" \
  -m main.eval_longbench_chunk2_heatmap \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --eval_data "${EVAL_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --result_dir "${RESULT_DIR}" \
  --figure_dir "${FIGURE_DIR}" \
  --figure_name "${FIGURE_NAME}" \
  --heatmap_vmax "${HEATMAP_VMAX}" \
  --beacon_probe_chunk_idx "${PROBE_CHUNK_IDX}" \
  --enable_beacon True \
  --attn_impl "${ATTN_IMPL}" \
  --chat_template "${CHAT_TEMPLATE}" \
  --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --max_samples_per_task "${MAX_SAMPLES_PER_TASK}" \
  --sample_seed "${SAMPLE_SEED}" \
  --sample_shuffle "${SAMPLE_SHUFFLE}" \
  --probe_max_new_tokens "${PROBE_MAX_NEW_TOKENS}" \
  --stop_after_first_heatmap "${STOP_AFTER_FIRST_HEATMAP}" \
  --beacon_ratio "${BEACON_RATIO}" \
  --beacon_ratio_mix "${BEACON_RATIO_MIX}" \
  --beacon_skip_rope_for_beacon_k "${BEACON_SKIP_ROPE_FOR_BEACON_K}" \
  --beacon_chunk_fixed_rope_for_beacon_k "${BEACON_CHUNK_FIXED_ROPE_FOR_BEACON_K}" \
  --beacon_enhance_k_pos_residual "${BEACON_ENHANCE_K_POS_RESIDUAL}" \
  --beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
  --beacon_headwise_up_proj "${BEACON_HEADWISE_UP_PROJ}" \
  --beacon_up_proj_group_size "${BEACON_UP_PROJ_GROUP_SIZE}" \
  "${EXTRA_COMPRESSION_ARGS[@]}" \
  "${TASK_ARGS[@]}"
