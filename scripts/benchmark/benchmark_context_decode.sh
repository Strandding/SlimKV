#!/usr/bin/env bash
# B1 BF16 FA2 comparison with shared cache and execution policies.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
if [[ "${1:-}" == "--help" ]]; then
  "${PYTHON:-python3}" -m tools.benchmark.benchmark_context_decode --help
  exit 0
fi
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a trained Llama SlimKV checkpoint.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
execution_mode="${EXECUTION_MODE:-optimized}"
read -r -a lengths <<< "${CONTEXT_LENGTHS:-32768 65536 98304 131072}"
read -r -a ratios <<< "${BEACON_RATIOS:-4 8}"
optional=()
if [[ -n "${ROUTES:-}" ]]; then
  read -r -a routes <<< "${ROUTES}"
  optional+=(--routes "${routes[@]}")
fi
case "${REFERENCE_CACHE_ON_CPU:-auto}" in
  auto) ;;
  True) optional+=(--reference_cache_on_cpu) ;;
  False) optional+=(--no-reference_cache_on_cpu) ;;
  *) echo "ERROR: REFERENCE_CACHE_ON_CPU must be auto, True or False." >&2; exit 2 ;;
esac
decode_suffix=""
if [[ "${DECODE_COMPRESS:-True}" == "False" ]]; then
  optional+=(--no-beacon_decode_compress)
  decode_suffix="_raw_decode"
fi
if [[ -n "${PROMPT_FILE:-}" ]]; then
  optional+=(--prompt_file "${PROMPT_FILE}")
fi
if [[ -n "${REFERENCE_DIR:-}" ]]; then
  optional+=(--reference_dir "${REFERENCE_DIR}")
fi
if [[ "${DECODE_COMPILE:-True}" == "False" ]]; then
  optional+=(--no-compile_tail)
fi
"${PYTHON:-python3}" -m tools.benchmark.benchmark_context_decode \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR:-data/results/context_decode_${execution_mode}${decode_suffix}}" \
  --execution_mode "${execution_mode}" \
  --context_lengths "${lengths[@]}" --beacon_ratios "${ratios[@]}" \
  --prefill_warmup "${PREFILL_WARMUP:-1}" \
  --decode_steps "${DECODE_STEPS:-128}" --warmup "${WARMUP:-1}" \
  --repetitions "${REPETITIONS:-4}" --seed "${SEED:-42}" \
  "${optional[@]}" "$@"
