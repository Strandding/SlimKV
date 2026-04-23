#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7,8,9}"
ROUNDS="${ROUNDS:-20}"
TENSOR_MB="${TENSOR_MB:-64}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-5}"

count_gpus() {
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

NPROC_PER_NODE="$(count_gpus "${GPU_LIST}")"
if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
  echo "ERROR: invalid GPU_LIST=${GPU_LIST}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

echo "=========================================="
echo " NCCL Communication Test"
echo "=========================================="
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  NPROC_PER_NODE       : ${NPROC_PER_NODE}"
echo "  ROUNDS               : ${ROUNDS}"
echo "  TENSOR_MB            : ${TENSOR_MB}"
echo "  WARMUP_ROUNDS        : ${WARMUP_ROUNDS}"
echo "=========================================="

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" \
  "${SCRIPT_DIR}/nccl_comm_test.py" \
  --rounds "${ROUNDS}" \
  --tensor-mb "${TENSOR_MB}" \
  --warmup-rounds "${WARMUP_ROUNDS}"
