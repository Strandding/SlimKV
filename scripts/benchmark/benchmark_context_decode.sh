#!/usr/bin/env bash
# Matched B1 BF16 FA2 decode; Graphs and compilation are disabled for all routes.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
if [[ "${1:-}" == "--help" ]]; then
  "${PYTHON:-python3}" -m tools.benchmark_context_decode --help
  exit 0
fi
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a trained Llama SlimKV checkpoint.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
read -r -a lengths <<< "${CONTEXT_LENGTHS:-32768 65536 98304 131072}"
read -r -a ratios <<< "${BEACON_RATIOS:-4 8}"
optional=()
if [[ -n "${PROMPT_FILE:-}" ]]; then
  optional+=(--prompt_file "${PROMPT_FILE}")
fi
"${PYTHON:-python3}" -m tools.benchmark_context_decode \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR:-data/results/context_decode}" \
  --context_lengths "${lengths[@]}" --beacon_ratios "${ratios[@]}" \
  --decode_steps "${DECODE_STEPS:-128}" --warmup "${WARMUP:-1}" \
  --repetitions "${REPETITIONS:-4}" --seed "${SEED:-42}" \
  "${optional[@]}" "$@"
