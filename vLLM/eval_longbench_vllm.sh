#!/usr/bin/env bash

# ================================================================
#  USER CONFIGURATION
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/longbench_vllm_models.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${MODEL_NAME_OR_PATH_DEFAULT}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${CHAT_TEMPLATE_DEFAULT}}"
DTYPE="${DTYPE:-${DTYPE_DEFAULT}}"
VLLM_TP="${VLLM_TP:-}"
VLLM_TP_AUTO="${VLLM_TP_AUTO:-True}"
VLLM_LEN="${VLLM_LEN:-${VLLM_LEN_DEFAULT}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VLLM_MEM="${VLLM_MEM:-0.9}"
VLLM_DISABLE_AR="${VLLM_DISABLE_AR:-False}"

EVAL_DATA="${EVAL_DATA:-/dataset/common/tzh/dataset/long-llm/longbench}"
RESULT_ROOT="${RESULT_ROOT:-data/results/longbench_vllm}"
RUN_TAG="${RUN_TAG:-${MODEL_PROFILE}_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RESULT_DIR:-${RUN_TAG}}"

MAX_LENGTH="${MAX_LENGTH:-31500}"
TRUNCATE_FROM_MIDDLE="${TRUNCATE_FROM_MIDDLE:-True}"
BATCH_SIZE="${BATCH_SIZE:-1}"

MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-42}"
EVAL_SAMPLE_SHUFFLE="${EVAL_SAMPLE_SHUFFLE:-True}"

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
  # Keep passage retrieval right after passage count.
  passage_retrieval_en
  lcc
  repobench-p
)

# Do not evaluate Chinese subsets.
ZH_TASKS=(
  multifieldqa_zh
  dureader
  vcsum
  lsht
  passage_retrieval_zh
)

# ================================================================
#  Internal logic
# ================================================================

set -euo pipefail

count_visible_gpus() {
  local raw="$1"
  local normalized
  normalized="$(echo "${raw}" | tr ',' ' ' | xargs)"
  if [[ -z "${normalized}" ]]; then
    echo 0
    return 0
  fi
  local count=0
  for _gpu in ${normalized}; do
    count=$((count + 1))
  done
  echo "${count}"
}

# Filter Chinese subsets defensively in case TASKS is edited externally.
filtered_tasks=()
for task in "${TASKS[@]}"; do
  is_zh_task=0
  for zh_task in "${ZH_TASKS[@]}"; do
    if [[ "${task}" == "${zh_task}" ]]; then
      is_zh_task=1
      break
    fi
  done
  if [[ "${is_zh_task}" -eq 0 ]]; then
    filtered_tasks+=("${task}")
  fi
done
TASKS=("${filtered_tasks[@]}")

NUM_VISIBLE_GPUS="$(count_visible_gpus "${CUDA_VISIBLE_DEVICES}")"
if [[ "${NUM_VISIBLE_GPUS}" -le 0 ]]; then
  echo "ERROR: Invalid CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

# Auto-select TP from visible GPUs by default.
# To force manual TP, set VLLM_TP_AUTO=False and pass VLLM_TP explicitly.
if [[ "${VLLM_TP_AUTO}" == "True" ]]; then
  VLLM_TP="${NUM_VISIBLE_GPUS}"
elif [[ -z "${VLLM_TP}" ]]; then
  VLLM_TP="${NUM_VISIBLE_GPUS}"
fi

if [[ "${VLLM_TP}" -gt "${NUM_VISIBLE_GPUS}" ]]; then
  echo "ERROR: VLLM_TP=${VLLM_TP} exceeds visible GPU count=${NUM_VISIBLE_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})" >&2
  exit 1
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_NAME_OR_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_DATA}/${TASKS[0]}.jsonl" ]]; then
  for p in \
    /dataset/common/tzh/dataset/long-llm/longbench \
    /dataset/common/tzh/datasets/long-llm/longbench \
    /dataset/common/tzh/.dataset/long-llm/longbench; do
    if [[ -f "${p}/${TASKS[0]}.jsonl" ]]; then
      EVAL_DATA="${p}"
      break
    fi
  done
fi

if [[ ! -f "${EVAL_DATA}/${TASKS[0]}.jsonl" ]]; then
  echo "ERROR: LongBench data not found. Please set EVAL_DATA." >&2
  exit 1
fi

eval_sample_args=()
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  eval_sample_args+=(--max_samples_per_task "${MAX_SAMPLES_PER_TASK}")
  eval_sample_args+=(--sample_seed "${EVAL_SAMPLE_SEED}")
  eval_sample_args+=(--sample_shuffle "${EVAL_SAMPLE_SHUFFLE}")
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=0
export MAP_NUM_PROC="${MAP_NUM_PROC:-32}"
export MAP_BATCH_SIZE="${MAP_BATCH_SIZE:-10}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "=========================================="
echo " LongBench Evaluation (vLLM)"
echo "=========================================="
echo "  PROFILE      : ${MODEL_PROFILE}"
echo "  MODEL        : ${MODEL_NAME_OR_PATH}"
echo "  EVAL_DATA    : ${EVAL_DATA}"
echo "  RESULT_DIR   : ${RESULT_ROOT}/${RESULT_DIR}"
echo "  CUDA_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  VLLM         : tp=${VLLM_TP}, mem=${VLLM_MEM}, max_model_len=${VLLM_LEN}, disable_ar=${VLLM_DISABLE_AR}"
echo "  RUNTIME      : chat_template=${CHAT_TEMPLATE}, dtype=${DTYPE}, batch=${BATCH_SIZE}, max_len=${MAX_LENGTH}, truncate_middle=${TRUNCATE_FROM_MIDDLE}"
if [[ -n "${MAX_SAMPLES_PER_TASK}" ]]; then
  echo "  SAMPLE_CAP   : per_task=${MAX_SAMPLES_PER_TASK}, shuffle=${EVAL_SAMPLE_SHUFFLE}, seed=${EVAL_SAMPLE_SEED}"
else
  echo "  SAMPLE_CAP   : full dataset"
fi
echo "  TASKS        : ${TASKS[*]}"
echo "=========================================="

python -m main.eval_longbench \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --eval_data "${EVAL_DATA}" \
  --output_dir "${RESULT_ROOT}" \
  --result_dir "${RESULT_DIR}" \
  --tasks "${TASKS[@]}" \
  --enable_vllm \
  --vllm_tp "${VLLM_TP}" \
  --vllm_mem "${VLLM_MEM}" \
  --vllm_len "${VLLM_LEN}" \
  --vllm_disable_ar "${VLLM_DISABLE_AR}" \
  --chat_template "${CHAT_TEMPLATE}" \
  --dtype "${DTYPE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --truncate_from_middle "${TRUNCATE_FROM_MIDDLE}" \
  "${eval_sample_args[@]}"

python "${SCRIPT_DIR}/summarize_longbench_scores.py" \
  --result_dir "${RESULT_ROOT}/${RESULT_DIR}" \
  --save_path "${RESULT_ROOT}/${RESULT_DIR}/score_summary.json"
