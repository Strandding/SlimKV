#!/usr/bin/env bash
# Model defaults for the shared LongBench launcher.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
DEFAULT_FINETUNE_DIR="${DEFAULT_FINETUNE_DIR:-${SLIMKV_OUTPUT_DIR}/slimkv-llama31-8b-finetune-dense}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
MASTER_PORT="${MASTER_PORT:-29531}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-llama31-8b-finetune-dense}"
BEACON_RATIO="${BEACON_RATIO:-8}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-llama-3}"
RAW_NOKROPE="${RAW_NOKROPE:-False}"
SLIMKV_EVAL_LABEL=Llama3.1-8B
source "${PROJECT_ROOT}/scripts/evaluation/longbench_common.sh" "$@"
