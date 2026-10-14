#!/usr/bin/env bash
# Llama3 SlimKV inference on LongBench-v1.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_DATA=/path/to/longbench \
       bash scripts/evaluation/eval_longbench_slimkv.sh [evaluator arguments]

Defaults: 15 English/code tasks, token ratio 4, BF16, final input limit 31500,
head/tail truncation, latent FA2 decode, CUDA Graphs, batch limit 32.
Feature ranks are read from the checkpoint; use an f4 checkpoint for t4f4.
Environment: PYTHON, CUDA_VISIBLE_DEVICES, OUTPUT_DIR, RESULT_DIR, TASKS,
BEACON_RATIO, BATCH_SIZE, MAX_BATCH_TOKENS, MAX_LENGTH, MAX_SAMPLES_PER_TASK,
DECODE_BACKEND, ATTN_IMPL, DECODE_GRAPH, LOAD_RESULT, MAP_NUM_PROC.
TASKS is a space-separated task list or 'english'/'all'.
HELP
  exit 0
fi
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a trained Llama3 SlimKV checkpoint.}"
EVAL_DATA="${EVAL_DATA:-${SLIMKV_DATA_ROOT}/longbench}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAP_NUM_PROC="${MAP_NUM_PROC:-8}"
export PYTHONUNBUFFERED=1
MAX_LENGTH="${MAX_LENGTH:-31500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BEACON_RATIO="${BEACON_RATIO:-4}"
DECODE_BACKEND="${DECODE_BACKEND:-fa2}"
if [[ "${DECODE_BACKEND}" == "fa2" ]]; then
  decode_fast=True
  graph_default=True
else
  decode_fast=False
  graph_default=False
fi
# The training corpus is Chinese-only, so the English preset excludes Chinese tasks.
# Passage counting is excluded separately; English retrieval and code tasks remain.
read -r -a tasks <<< "${TASKS:-english}"
sample_args=()
if [[ -n "${MAX_SAMPLES_PER_TASK:-}" ]]; then
  sample_args+=(--max_samples_per_task "${MAX_SAMPLES_PER_TASK}")
fi
"${PYTHON:-python3}" -m main.eval_longbench \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --eval_data "${EVAL_DATA}" \
  --output_dir "${OUTPUT_DIR:-data/results/longbench}" \
  --result_dir "${RESULT_DIR:-slimkv-t${BEACON_RATIO}}" \
  --tasks "${tasks[@]}" \
  --enable_beacon True --dtype bf16 --chat_template llama-3 \
  --beacon_ratio "${BEACON_RATIO}" --beacon_ratio_mix sequence \
  --beacon_skip_rope_for_beacon_k True \
  --beacon_chunk_fixed_rope_for_beacon_k False \
  --beacon_cache_mode latent --beacon_decode_backend "${DECODE_BACKEND}" \
  --beacon_decode_fast "${decode_fast}" \
  --beacon_decode_graph "${DECODE_GRAPH:-${graph_default}}" \
  --attn_impl "${ATTN_IMPL:-flash_attention_2}" \
  --max_length "${MAX_LENGTH}" --truncate_from_middle True --truncate_after_chat True \
  --batch_size "${BATCH_SIZE}" --group_by_input_length True \
  --max_batch_tokens "${MAX_BATCH_TOKENS:-$((BATCH_SIZE * MAX_LENGTH))}" \
  --num_logits_to_keep 1 --do_sample False --sample_shuffle False \
  --load_result "${LOAD_RESULT:-False}" \
  "${sample_args[@]}" "$@"
