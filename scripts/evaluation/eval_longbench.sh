#!/usr/bin/env bash
# Model defaults for the shared LongBench launcher.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${SLIMKV_OUTPUT_DIR}/slimkv-qwen2-finetune-lowrank-nokrope-shared-down-kv64/checkpoint-3000}"
MASTER_PORT="${MASTER_PORT:-29501}"
OUTPUT_NAME="${OUTPUT_NAME:-slimkv-qwen2-finetune-lowrank-kv64}"
BEACON_RATIO="${BEACON_RATIO:-16}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
RAW_NOKROPE=False
SLIMKV_EVAL_LABEL=Qwen
source "${PROJECT_ROOT}/scripts/evaluation/longbench_common.sh" "$@"
