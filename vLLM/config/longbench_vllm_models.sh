#!/usr/bin/env bash

# LongBench + vLLM model profiles.
# Add a new profile in the case block below when you want to test another model.
#
# Exported defaults:
#   MODEL_NAME_OR_PATH_DEFAULT
#   CHAT_TEMPLATE_DEFAULT
#   DTYPE_DEFAULT
#   VLLM_TP_DEFAULT
#   VLLM_LEN_DEFAULT

set -euo pipefail

MODEL_PROFILE="${MODEL_PROFILE:-qwen25_14b}"

case "${MODEL_PROFILE}" in
  qwen25_14b)
    MODEL_NAME_OR_PATH_DEFAULT="/dataset/common/tzh/.model/Qwen2.5-14B-Instruct"
    CHAT_TEMPLATE_DEFAULT="qwen"
    DTYPE_DEFAULT="bf16"
    VLLM_TP_DEFAULT="2"
    VLLM_LEN_DEFAULT="32768"
    ;;
  *)
    echo "ERROR: Unknown MODEL_PROFILE=${MODEL_PROFILE}" >&2
    echo "Please edit config/longbench_vllm_models.sh and add this profile." >&2
    exit 1
    ;;
esac
