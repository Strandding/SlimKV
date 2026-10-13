#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
set -euo pipefail

# Monitor GPU memory usage and trigger cache occupy as soon as any monitored GPU
# memory usage is below threshold.

GPU_IDS="${GPU_IDS:-}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-3}"
THRESHOLD_MB="${THRESHOLD_MB:-50000}"   # 50 GB ~= 50000 MiB
OCCUPY_MEM_GB="${OCCUPY_MEM_GB:-}"       # empty => dynamic fill
RESERVE_MB="${RESERVE_MB:-512}"          # used only in dynamic fill mode

if [[ -n "${GPU_IDS}" ]]; then
  QUERY_CMD=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_IDS}")
  TARGET_DESC="${GPU_IDS}"
else
  QUERY_CMD=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  TARGET_DESC="all"
fi

if [[ -n "${OCCUPY_MEM_GB}" ]]; then
  echo "[watch] gpu_ids=${TARGET_DESC} interval=${CHECK_INTERVAL_SEC}s threshold=${THRESHOLD_MB}MiB occupy_mem=${OCCUPY_MEM_GB}GB"
else
  echo "[watch] gpu_ids=${TARGET_DESC} interval=${CHECK_INTERVAL_SEC}s threshold=${THRESHOLD_MB}MiB occupy_mem=dynamic(reserve=${RESERVE_MB}MB)"
fi
echo "[watch] waiting for any monitored GPU memory.used < ${THRESHOLD_MB}MiB ..."

while true; do
  # Returns one line per monitored GPU.
  mapfile -t used_mb < <("${QUERY_CMD[@]}")

  trigger=0
  for v in "${used_mb[@]}"; do
    if [[ "${v}" -lt "${THRESHOLD_MB}" ]]; then
      trigger=1
      break
    fi
  done

  if [[ "${trigger}" -eq 1 ]]; then
    echo "[watch] trigger at $(date '+%F %T'): some GPU < ${THRESHOLD_MB}MiB. start cache occupy now."
    # Occupy with compute enabled (default). This call is blocking.
    cache_cmd=(python -m tools.gpu_occupy)
    if [[ -n "${OCCUPY_MEM_GB}" ]]; then
      cache_cmd+=(--mem "${OCCUPY_MEM_GB}")
    else
      cache_cmd+=(--reserve-mb "${RESERVE_MB}")
    fi

    if [[ -n "${GPU_IDS}" ]]; then
      exec env CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${cache_cmd[@]}"
    else
      exec "${cache_cmd[@]}"
    fi
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
