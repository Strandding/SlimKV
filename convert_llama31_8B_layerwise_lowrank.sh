#!/usr/bin/env bash

# Run this script on a compute node.
# It converts a dense-beacon Llama3.1-8B checkpoint to low-rank beacon K/V
# using per-layer latent dims (K/V can be different).
# This script uses:
#   - global SVD partition (NOT per-head SVD)
#   - non-headwise up-proj (NOT grouped up-proj)

PYTHON_BIN="${PYTHON_BIN:-/export/home2/zihan/miniconda3/envs/beacon/bin/python}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/dataset/common/tzh/.model/Llama-3.1-8B-Instruct}"
OUTPUT_NAME="${OUTPUT_NAME:-beacon-llama31-8b-lowrank-kada-v64}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/${OUTPUT_NAME}}"

DTYPE="${DTYPE:-bf16}"
SAVE_DTYPE="${SAVE_DTYPE:-bf16}"
SVD_DEVICE="${SVD_DEVICE:-cuda}"
SVD_PARTITION="${SVD_PARTITION:-global}"
SAFE_SERIALIZATION="${SAFE_SERIALIZATION:-true}"
SAVE_SVD_SPECTRA="${SAVE_SVD_SPECTRA:-true}"
SAVE_SVD_SPECTRA_DATA="${SAVE_SVD_SPECTRA_DATA:-true}"

BEACON_SKIP_ROPE_FOR_BEACON_K="${BEACON_SKIP_ROPE_FOR_BEACON_K:-True}"
BEACON_SHARED_KV_DOWN_PROJ="${BEACON_SHARED_KV_DOWN_PROJ:-False}"
BEACON_HEADWISE_UP_PROJ="${BEACON_HEADWISE_UP_PROJ:-False}"

BEACON_K_LATENT_HEAD_DIM_LIST="${BEACON_K_LATENT_HEAD_DIM_LIST:-48 48 48 48 52 52 56 56 60 60 64 64 68 68 72 72 72 72 76 76 80 80 80 80 76 76 72 72 64 64 56 56}"
BEACON_V_LATENT_HEAD_DIM_LIST="${BEACON_V_LATENT_HEAD_DIM_LIST:-64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64}"

read -r -a K_LAYERWISE_DIMS <<< "${BEACON_K_LATENT_HEAD_DIM_LIST}"
read -r -a V_LAYERWISE_DIMS <<< "${BEACON_V_LATENT_HEAD_DIM_LIST}"
if (( ${#K_LAYERWISE_DIMS[@]} != 32 )); then
  echo "Invalid BEACON_K_LATENT_HEAD_DIM_LIST length: ${#K_LAYERWISE_DIMS[@]} (expected 32 for Llama3.1-8B)." >&2
  exit 1
fi
if (( ${#V_LAYERWISE_DIMS[@]} != 32 )); then
  echo "Invalid BEACON_V_LATENT_HEAD_DIM_LIST length: ${#V_LAYERWISE_DIMS[@]} (expected 32 for Llama3.1-8B)." >&2
  exit 1
fi
for dim in "${K_LAYERWISE_DIMS[@]}"; do
  if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid K latent dim in list: ${dim}" >&2
    exit 1
  fi
done
for dim in "${V_LAYERWISE_DIMS[@]}"; do
  if ! [[ "${dim}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid V latent dim in list: ${dim}" >&2
    exit 1
  fi
done

echo "[convert_llama31_8B_layerwise_lowrank] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[convert_llama31_8B_layerwise_lowrank] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[convert_llama31_8B_layerwise_lowrank] DTYPE=${DTYPE} SAVE_DTYPE=${SAVE_DTYPE}"
echo "[convert_llama31_8B_layerwise_lowrank] SVD_DEVICE=${SVD_DEVICE} SVD_PARTITION=${SVD_PARTITION}"
echo "[convert_llama31_8B_layerwise_lowrank] HEADWISE_UP=${BEACON_HEADWISE_UP_PROJ}"
echo "[convert_llama31_8B_layerwise_lowrank] SHARED_DOWN=${BEACON_SHARED_KV_DOWN_PROJ}"
echo "[convert_llama31_8B_layerwise_lowrank] K_LATENT_DIMS=${BEACON_K_LATENT_HEAD_DIM_LIST}"
echo "[convert_llama31_8B_layerwise_lowrank] V_LATENT_DIMS=${BEACON_V_LATENT_HEAD_DIM_LIST}"

"${PYTHON_BIN}" -m main.convert_beacon_kv_lowrank \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --dtype "${DTYPE}" \
  --save_dtype "${SAVE_DTYPE}" \
  --svd_device "${SVD_DEVICE}" \
  --svd_partition "${SVD_PARTITION}" \
  --safe_serialization "${SAFE_SERIALIZATION}" \
  --save_svd_spectra "${SAVE_SVD_SPECTRA}" \
  --save_svd_spectra_data "${SAVE_SVD_SPECTRA_DATA}" \
  --enable_beacon \
  --beacon_param q k v \
  --beacon_skip_rope_for_beacon_k "${BEACON_SKIP_ROPE_FOR_BEACON_K}" \
  --beacon_shared_kv_down_proj "${BEACON_SHARED_KV_DOWN_PROJ}" \
  --beacon_headwise_up_proj "${BEACON_HEADWISE_UP_PROJ}" \
  --beacon_k_latent_head_dim_list "${K_LAYERWISE_DIMS[@]}" \
  --beacon_v_latent_head_dim_list "${V_LAYERWISE_DIMS[@]}"
